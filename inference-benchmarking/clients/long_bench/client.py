import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tiktoken
import torch.multiprocessing as mp
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class Config:
    save_dir: str = "results"
    model_name: str = "GLM-4-9B-Chat"
    tokenizer_path: str = None
    max_seq_len: int = 120000
    cot: bool = False
    no_context: bool = False
    rag: int = 0
    n_proc: int = 16
    limit: int = None


class LongBenchClient:
    def __init__(self):
        self.logger = logging.getLogger("LongBenchClient")
        self.config = None
        self.load_templates()

    def setup(self):
        pass

    def load_templates(self):
        """Load prompt templates"""
        templates_dir = Path(__file__).parent / "prompts"
        try:
            self.templates = {
                "rag": (templates_dir / "0shot_rag.txt").read_text(encoding="utf-8"),
                "no_context": (templates_dir / "0shot_no_context.txt").read_text(encoding="utf-8"),
                "0shot": (templates_dir / "0shot.txt").read_text(encoding="utf-8"),
                "cot": (templates_dir / "0shot_cot.txt").read_text(encoding="utf-8"),
                "cot_ans": (templates_dir / "0shot_cot_ans.txt").read_text(encoding="utf-8"),
            }
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load templates from {templates_dir}: {e}")
            raise

    def evaluate(
        self,
        model_path: str,
        server_port: int,
        results_dir: str,
        model_name: str = None,
        max_concurrent_requests: int = 1,
        timeout: int = 3600,
        max_seq_len: int = 120000,
        use_cot: bool = False,
        no_context: bool = False,
        rag: int = 0,
        limit: int = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], str]:
        """Run LongBench evaluation"""
        self.logger.info(f"Running LongBench evaluation for model {model_name}")

        os.makedirs(results_dir, exist_ok=True)

        # Set environment variables for the API
        os.environ["URL"] = f"http://localhost:{server_port}/v1"
        os.environ["API_KEY"] = "EMPTY"

        try:
            self.config = Config(
                save_dir=results_dir,
                model_name=model_path,
                tokenizer_path=model_path,
                max_seq_len=max_seq_len,
                cot=use_cot,
                no_context=no_context,
                rag=rag,
                n_proc=max_concurrent_requests,
                limit=limit,
            )

            self.pred_main()

            results_file = self.out_file

            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = [json.loads(line) for line in f]
                return self._process_results(results), results_file
            else:
                raise FileNotFoundError(f"Results file not found: {results_file}")

        except Exception as e:
            self.logger.error(f"Error during LongBench evaluation: {str(e)}")
            raise

    def query_llm(
        self,
        prompt: str,
        tokenizer,
        client: OpenAI,
        temperature: float = 0.5,
        max_new_tokens: int = 128,
    ) -> str:
        """Query LLM with truncation if needed"""
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > self.config.max_seq_len:
            input_ids = (
                input_ids[: self.config.max_seq_len // 2]
                + input_ids[-self.config.max_seq_len // 2 :]
            )
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

        tries = 0
        # model = f"models/{self.config.model_name}/"
        model = self.config.model_name  # Use the full path directly
        while tries < 5:
            tries += 1
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                return completion.choices[0].message.content
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.warning(f"Error in API call: {str(e)}. Retry {tries}/5")
                time.sleep(1)
        return ""

    def pred_main(self):
        """Main prediction loop"""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Determine output file name
        suffix = ""
        if self.config.rag > 0:
            suffix = f"_rag_{self.config.rag}"
        elif self.config.no_context:
            suffix = "_no_context"
        elif self.config.cot:
            suffix = "_cot"

        model_name = os.path.basename(self.config.model_name.strip("/"))
        self.out_file = Path(self.config.save_dir) / f"{model_name}{suffix}.jsonl"

        # Load dataset
        dataset = load_dataset("THUDM/LongBench-v2", split="train")
        data_all = [
            {
                k: item[k]
                for k in [
                    "_id",
                    "domain",
                    "sub_domain",
                    "difficulty",
                    "length",
                    "question",
                    "choice_A",
                    "choice_B",
                    "choice_C",
                    "choice_D",
                    "answer",
                    "context",
                ]
            }
            for item in dataset
        ]

        # Handle existing results
        has_data = set()
        if self.out_file.exists():
            with open(self.out_file, encoding="utf-8") as f:
                has_data = {json.loads(line)["_id"] for line in f}

        data = [item for item in data_all if item["_id"] not in has_data]

        # Apply limit early if specified
        if self.config.limit is not None:
            self.logger.info(f"Limiting data to {self.config.limit} items out of {len(data)}")
            data = data[: self.config.limit]

        with open(self.out_file, "a", encoding="utf-8") as fout:
            if self.config.n_proc > 1:
                # Parallel processing
                data_subsets = [data[i :: self.config.n_proc] for i in range(self.config.n_proc)]
                processes = []
                for subset in data_subsets:
                    p = mp.Process(target=self.get_pred, args=(subset, fout))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                # Single process
                self.get_pred(data, fout)

    def extract_answer(self, response: str) -> str:
        """Extract answer from response"""
        response = response.replace("*", "")
        match = re.search(r"The correct answer is \(([A-D])\)", response)
        if match:
            return match.group(1)
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        return None

    def get_pred(self, data: List[Dict], fout):
        """Process predictions for a subset of data"""
        if "gpt" in self.config.model_name or "o1" in self.config.model_name:
            tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path, trust_remote_code=True
            )

        client = OpenAI(base_url=os.environ["URL"], api_key=os.environ["API_KEY"])

        for item in tqdm(data):
            context = item["context"]
            if self.config.rag > 0:
                template = self.templates["rag"]
                retrieved = item["retrieved_context"][: self.config.rag]
                retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
                context = "\n\n".join(
                    [f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)]
                )
            elif self.config.no_context:
                template = self.templates["no_context"]
            elif self.config.cot:
                template = self.templates["cot"]
            else:
                template = self.templates["0shot"]

            prompt = template.replace("$DOC$", context.strip())
            for field in ["question", "choice_A", "choice_B", "choice_C", "choice_D"]:
                prompt = prompt.replace(f"${field.upper()}$", item[field].strip())

            max_new_tokens = 1024 if self.config.cot else 128
            output = self.query_llm(
                prompt, tokenizer, client, temperature=0.1, max_new_tokens=max_new_tokens
            )
            if not output:
                continue

            if self.config.cot:
                response = output.strip()
                item["response_cot"] = response
                prompt = self.templates["cot_ans"].replace("$COT$", response)
                for field in ["DOC", "question", "choice_A", "choice_B", "choice_C", "choice_D"]:
                    val = context if field == "DOC" else item[field.lower()]
                    prompt = prompt.replace(f"${field}$", val.strip())

                output = self.query_llm(
                    prompt, tokenizer, client, temperature=0.1, max_new_tokens=128
                )
                if not output:
                    continue

            response = output.strip()
            item["response"] = response
            item["pred"] = self.extract_answer(response)
            item["judge"] = item["pred"] == item["answer"]
            item["context"] = context[:1000]  # Truncate context for storage
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()

    def _get_results_file(
        self, save_dir: str, model_name: str, use_cot: bool, no_context: bool, rag: int
    ) -> str:
        """Get the path to results file based on configuration"""
        # base_name = model_name.split("/")[-1]
        if rag > 0:
            suffix = f"_rag_{rag}"
        elif no_context:
            suffix = "_no_context"
        elif use_cot:
            suffix = "_cot"
        else:
            suffix = ""

        return os.path.join(save_dir, f"{model_name}{suffix}.jsonl")

    def _process_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Process and summarize results"""
        summary = {
            "total_samples": len(results),
            "correct": sum(1 for r in results if r.get("judge", False)),
            "by_domain": defaultdict(lambda: {"total": 0, "correct": 0}),
            "by_difficulty": defaultdict(lambda: {"total": 0, "correct": 0}),
        }

        for result in results:
            domain = result.get("domain", "unknown")
            difficulty = result.get("difficulty", "unknown")
            is_correct = result.get("judge", False)

            summary["by_domain"][domain]["total"] += 1
            summary["by_domain"][domain]["correct"] += int(is_correct)

            summary["by_difficulty"][difficulty]["total"] += 1
            summary["by_difficulty"][difficulty]["correct"] += int(is_correct)

        # Calculate accuracies
        summary["accuracy"] = summary["correct"] / summary["total_samples"]

        for domain in summary["by_domain"]:
            d = summary["by_domain"][domain]
            d["accuracy"] = d["correct"] / d["total"]

        for diff in summary["by_difficulty"]:
            d = summary["by_difficulty"][diff]
            d["accuracy"] = d["correct"] / d["total"]

        return summary

    def eval_results(self, save_dir):
        files = os.listdir(save_dir)
        output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"]
        compensated = False

        for file in files:
            filename = os.path.join("results", file)
            try:
                pred_data = json.load(open(filename, encoding="utf-8"))
            except Exception as e:
                pred_data = [json.loads(line) for line in open(filename, encoding="utf-8")]
            easy, hard, short, medium, long = 0, 0, 0, 0, 0
            easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
            for pred in pred_data:
                acc = int(pred["judge"])
                if compensated and pred["pred"] == None:
                    acc = 0.25
                if pred["difficulty"] == "easy":
                    easy += 1
                    easy_acc += acc
                else:
                    hard += 1
                    hard_acc += acc

                if pred["length"] == "short":
                    short += 1
                    short_acc += acc
                elif pred["length"] == "medium":
                    medium += 1
                    medium_acc += acc
                else:
                    long += 1
                    long_acc += acc

            name = ".".join(file.split(".")[:-1])
            output.append(
                name
                + "\t"
                + str(round(100 * (easy_acc + hard_acc) / len(pred_data), 1))
                + "\t"
                + str(round(100 * easy_acc / easy, 1))
                + "\t"
                + str(round(100 * hard_acc / hard, 1))
                + "\t"
                + str(round(100 * short_acc / short, 1))
                + "\t"
                + str(round(100 * medium_acc / medium, 1))
                + "\t"
                + str(round(100 * long_acc / long, 1))
            )

        open("result.txt", "w", encoding="utf-8").write("\n".join(output))
