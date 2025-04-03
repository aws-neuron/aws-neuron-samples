import os
import re
import subprocess
import time

from botocore.utils import IMDSFetcher


class S3Utils:
    @staticmethod
    def get_instance_region():
        # IMDSv2 implementation of fetch az
        instance_region = (
            IMDSFetcher()
            ._get_request(
                "/latest/meta-data/placement/region",
                None,
                token=IMDSFetcher()._fetch_metadata_token(),
            )
            .text
        )

        # remove everything after 3rd "-" to remove local zone info and get region
        instance_region = "-".join(instance_region.split("-")[:3])

        return instance_region

    @staticmethod
    def is_s3_crt_supported(region):
        # CRT is only supported for in-region download
        in_region_buckets = ["us-east-1", "us-east-2", "us-west-2"]
        return region in in_region_buckets

    @staticmethod
    def enable_s3_crt():
        print(f"Enabling CRT transfer client for S3 downloads")
        s3_crt_config = {"s3.preferred_transfer_client": "crt", "s3.max_bandwidth": "100GB/s"}
        results = []
        for var in s3_crt_config.keys():
            enable_s3_crt_cmd = ["aws", "configure", "set", var, s3_crt_config[var]]
            result = subprocess.run(enable_s3_crt_cmd, capture_output=True, text=True)
            results.append(result)
        return results

    @staticmethod
    def disable_s3_crt():
        print("Disabling CRT transfer client for S3 downloads")
        disable_s3_crt_cmd = [
            "aws",
            "configure",
            "set",
            "default.s3.preferred_transfer_client",
            "classic",
        ]
        result = subprocess.run(disable_s3_crt_cmd, capture_output=True, text=True)
        return result

    @staticmethod
    def regionalize_s3_uri(s3_dir, region):
        # load models from kaena-nn-models bucket for us-west-2 and kaena-nn-models-{region} for other regions
        if "kaena-nn-models" in s3_dir and region not in s3_dir and region != "us-west-2":
            s3_dir = s3_dir.replace("kaena-nn-models", f"kaena-nn-models-{region}")
            print(f"Updating s3_dir to {s3_dir}")
        return s3_dir

    @staticmethod
    def get_dir_size(dir):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    @staticmethod
    def download_from_s3(s3_dir, local_dir, use_crt_if_available=True):
        # get instance region; update environment var to use region rather than local zone
        instance_region = get_instance_region()
        region_env_var = os.environ.get("AWS_REGION", None)
        if region_env_var != instance_region:
            print(f"Overriding AWS_REGION from {region_env_var} to {instance_region}")
            os.environ["AWS_REGION"] = instance_region

        # configure CRT
        allow_crt = False
        if is_s3_crt_supported(instance_region) and use_crt_if_available:
            allow_crt = True
            print(
                f"Model bucket exists in {instance_region}, will enable CRT transfer client for S3 downloads"
            )
            enable_s3_crt()
            s3_dir = regionalize_s3_uri(s3_dir, instance_region)
        else:
            print(
                f"Model bucket does not exist in {instance_region}, will use classic transfer client for S3 download"
            )
            disable_s3_crt()

        # track download time and size
        start_time = time.time()
        start_size = get_dir_size(local_dir)

        # download weights from s3
        sync_cmd = [
            "aws",
            "s3",
            "sync",
            s3_dir,
            local_dir,
            "--only-show-errors",
            "--exact-timestamps",
        ]
        try:
            print(f"Downloading from s3 <{s3_dir}> to local <{local_dir}>")
            result = subprocess.run(sync_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"{result.stderr}")

        except Exception as e:

            # retry without CRT if CRT fails
            if allow_crt:
                start_time = time.time()
                disable_s3_crt()
                print(f"Weight download from s3 failed: {e}")
                print(f"Retrying download without CRT from s3 <{s3_dir}> to local <{local_dir}>")
                result = subprocess.run(sync_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    raise Exception(f"{result.stderr}")

            else:
                raise Exception(f"Weight download from s3 failed: {e}")

        # leaving CRT enabled breaks other reads/writes in inference test flow
        finally:
            if allow_crt:
                disable_s3_crt()

        # calculate download statistics
        total_time = time.time() - start_time
        downloaded_bytes = get_dir_size(local_dir) - start_size
        print(f"Download duration: {round(total_time, 2)}s")
        print(f"Download size: {round(downloaded_bytes / 1000000000, 2)}GB")
        print(f"Download speed: {round(downloaded_bytes * 8 / 1000000000 / total_time, 2)}Gbps")


# Make commonly used functions available at module level
download_from_s3 = S3Utils.download_from_s3
get_instance_region = S3Utils.get_instance_region
is_s3_crt_supported = S3Utils.is_s3_crt_supported
enable_s3_crt = S3Utils.enable_s3_crt
regionalize_s3_uri = S3Utils.regionalize_s3_uri
disable_s3_crt = S3Utils.disable_s3_crt
get_dir_size = S3Utils.get_dir_size
