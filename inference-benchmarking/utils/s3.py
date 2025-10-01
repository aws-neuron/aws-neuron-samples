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
        print(f"Downloading from s3 <{s3_dir}> to local <{local_dir}>")
        result = subprocess.run(sync_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Weight download from s3 failed: {result.stderr}")

        # calculate download statistics
        total_time = time.time() - start_time
        downloaded_bytes = get_dir_size(local_dir) - start_size
        print(f"Download duration: {round(total_time, 2)}s")
        print(f"Download size: {round(downloaded_bytes / 1000000000, 2)}GB")
        print(f"Download speed: {round(downloaded_bytes * 8 / 1000000000 / total_time, 2)}Gbps")


# Make commonly used functions available at module level
download_from_s3 = S3Utils.download_from_s3
get_instance_region = S3Utils.get_instance_region
get_dir_size = S3Utils.get_dir_size
