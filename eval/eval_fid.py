# pip3 install clean-fid
# https://github.com/GaParmar/clean-fid
from cleanfid import fid
import argparse

def main(real_image_path, generated_image_path):
    score = 0.0
    # We have 4 groups of generated images
    for i in range(4):
        score += fid.compute_fid(
            real_image_path,
            f'{generated_image_path}/group_{i}',
            dataset_res=512,
            batch_size=128
        )
    # Report the average FID score
    print(score / 4)

if __name__ == "__main__":
    # For real images, you should load our huggingface datasets and then save each image into local path.
    # For generated images, you should run our evaluate sctipts for each condition.
    # Make sure the real images and the generated images have the same file name.
    parser = argparse.ArgumentParser(description="Compute FID score between real and generated images.")
    parser.add_argument('--real_image_path', type=str, required=True, help='Path to the real images.')
    parser.add_argument('--generated_image_path', type=str, required=True, help='Path to the generated images.')

    args = parser.parse_args()
    main(args.real_image_path, args.generated_image_path)