#!/usr/bin/env python
"""
Download and prepare CelebA dataset for training.
This script handles the complete dataset setup including download, extraction, and verification.
"""

import os
import zipfile
import gdown
import requests
from tqdm import tqdm
import hashlib
import argparse
from typing import Optional, Dict, List
import shutil
import sys


class CelebADownloader:
    """Download and extract CelebA dataset with progress tracking and verification."""
    
    # Google Drive IDs for CelebA files (may have quota issues)
    DRIVE_IDS = {
        'img_align_celeba.zip': '1cNIac61PSA_LqDFYFUeyaQYekYPc75NH',
        'list_eval_partition.txt': '1aNsQA7nMgoN8q9EprLxNFqJFw6BQuVVd',
        'identity_CelebA.txt': '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
        'list_attr_celeba.txt': '1xm6vqHQ-0d1sch7bLUMqOEqmkqVCrWBW',
        'list_bbox_celeba.txt': '1tLqMsAhFUkLSIJu3gYiUMFXrDUrmJeaH',
        'list_landmarks_align_celeba.txt': '1xDILQ1tHwiYYL2fYMslmVvBwHp8VBsVF'
    }
    
    # Alternative download methods when Google Drive fails
    ALTERNATIVE_SOURCES = {
        'kaggle': {
            'dataset': 'jessicali9530/celeba-dataset',
            'files': ['img_align_celeba.zip', 'list_eval_partition.txt', 'list_attr_celeba.txt']
        },
        'torchvision': True  # Can use torchvision.datasets.CelebA
    }
    
    # Expected file sizes for verification (in bytes)
    EXPECTED_SIZES = {
        'img_align_celeba.zip': 1443490838,  # ~1.4GB
        'list_eval_partition.txt': 446409,
        'identity_CelebA.txt': 3994854,
        'list_attr_celeba.txt': 26645026,
        'list_bbox_celeba.txt': 6169307,
        'list_landmarks_align_celeba.txt': 10486120
    }
    
    # MD5 checksums for verification (optional, but recommended)
    MD5_CHECKSUMS = {
        'img_align_celeba.zip': '00566ae06ac5b0d8e67e5c5cb7792aa4',
        'list_eval_partition.txt': 'd32c9cbf5e040fd66e47c5a0e3e8cf8d',
    }
    
    def __init__(self, data_dir: str = './assets/datasets/celeba', quiet: bool = False):
        """
        Initialize CelebA downloader.
        
        Args:
            data_dir: Directory to save the dataset
            quiet: If True, suppress progress output
        """
        self.data_dir = data_dir
        self.quiet = quiet
        os.makedirs(data_dir, exist_ok=True)
        
        # Check for required packages
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are installed."""
        try:
            import gdown
        except ImportError:
            print("Error: gdown package is required for downloading from Google Drive")
            print("Please install it with: pip install gdown")
            sys.exit(1)
    
    def download_from_gdrive(self, file_name: str, force: bool = False) -> bool:
        """
        Download file from Google Drive using gdown.
        
        Args:
            file_name: Name of the file to download
            force: Force re-download even if file exists
            
        Returns:
            True if successful, False otherwise
        """
        if file_name not in self.DRIVE_IDS:
            print(f"Unknown file: {file_name}")
            return False
        
        file_id = self.DRIVE_IDS[file_name]
        output_path = os.path.join(self.data_dir, file_name)
        
        # Check if file already exists
        if os.path.exists(output_path) and not force:
            if self.verify_file(file_name):
                if not self.quiet:
                    print(f"✓ {file_name} already exists and is valid")
                return True
            else:
                print(f"⚠ {file_name} exists but is corrupted, re-downloading...")
                os.remove(output_path)
        
        if not self.quiet:
            print(f"📥 Downloading {file_name}...")
            if file_name == 'img_align_celeba.zip':
                print("  This is a large file (~1.4GB) and may take several minutes...")
        
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            # Use gdown with progress bar
            gdown.download(url, output_path, quiet=self.quiet)
            
            # Verify download
            if self.verify_file(file_name):
                if not self.quiet:
                    print(f"✓ Successfully downloaded {file_name}")
                return True
            else:
                print(f"✗ Download of {file_name} failed verification")
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False
                
        except Exception as e:
            print(f"✗ Error downloading {file_name}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def verify_file(self, file_name: str, check_md5: bool = False) -> bool:
        """
        Verify downloaded file size and optionally MD5 checksum.
        
        Args:
            file_name: Name of the file to verify
            check_md5: Whether to verify MD5 checksum (slower but more accurate)
            
        Returns:
            True if file is valid, False otherwise
        """
        file_path = os.path.join(self.data_dir, file_name)
        
        if not os.path.exists(file_path):
            return False
        
        # Check file size
        actual_size = os.path.getsize(file_path)
        expected_size = self.EXPECTED_SIZES.get(file_name)
        
        if expected_size and abs(actual_size - expected_size) > 1000:  # Allow 1KB tolerance
            if not self.quiet:
                print(f"  Size mismatch for {file_name}:")
                print(f"    Expected: {expected_size:,} bytes")
                print(f"    Got: {actual_size:,} bytes")
            return False
        
        # Optionally check MD5
        if check_md5 and file_name in self.MD5_CHECKSUMS:
            if not self.quiet:
                print(f"  Verifying MD5 checksum for {file_name}...")
            
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5.update(chunk)
            
            actual_md5 = md5.hexdigest()
            expected_md5 = self.MD5_CHECKSUMS[file_name]
            
            if actual_md5 != expected_md5:
                if not self.quiet:
                    print(f"  MD5 mismatch for {file_name}")
                return False
        
        return True
    
    def extract_zip(self, zip_name: str, remove_after: bool = False) -> bool:
        """
        Extract zip file with progress bar.
        
        Args:
            zip_name: Name of the zip file to extract
            remove_after: Whether to remove zip file after extraction
            
        Returns:
            True if successful, False otherwise
        """
        zip_path = os.path.join(self.data_dir, zip_name)
        
        if not os.path.exists(zip_path):
            print(f"✗ Zip file not found: {zip_path}")
            return False
        
        # Check if already extracted
        extract_dir = os.path.join(self.data_dir, 'img_align_celeba')
        if os.path.exists(extract_dir):
            num_images = len([f for f in os.listdir(extract_dir) if f.endswith('.jpg')])
            if num_images == 202599:  # Expected number of images
                if not self.quiet:
                    print(f"✓ Images already extracted ({num_images:,} images found)")
                return True
            elif num_images > 0:
                print(f"⚠ Partial extraction found ({num_images:,} images), re-extracting...")
                shutil.rmtree(extract_dir)
        
        if not self.quiet:
            print(f"📦 Extracting {zip_name}...")
            print("  This may take a few minutes...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total number of files
                members = zip_ref.namelist()
                total_files = len(members)
                
                # Extract with progress bar
                with tqdm(total=total_files, desc="Extracting", disable=self.quiet) as pbar:
                    for member in members:
                        zip_ref.extract(member, self.data_dir)
                        pbar.update(1)
            
            if not self.quiet:
                print(f"✓ Successfully extracted {total_files:,} files")
            
            # Optionally remove zip file to save space
            if remove_after:
                os.remove(zip_path)
                if not self.quiet:
                    print(f"  Removed {zip_name} to save disk space")
            
            return True
            
        except Exception as e:
            print(f"✗ Error extracting {zip_name}: {e}")
            return False
    
    def download_with_torchvision(self) -> bool:
        """
        Alternative: Download CelebA using torchvision (more reliable).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import torchvision
            from torchvision.datasets import CelebA
            
            print("Using torchvision to download CelebA (more reliable)...")
            print("This will download the dataset in a different format.")
            print("Note: This may take 10-20 minutes for the first download.")
            
            # Download using torchvision
            dataset = CelebA(
                root=os.path.dirname(self.data_dir),  # Parent directory
                split='all',
                download=True
            )
            
            print(f"✓ Successfully downloaded {len(dataset)} images using torchvision")
            
            # Move files to expected location if needed
            torchvision_celeba = os.path.join(os.path.dirname(self.data_dir), 'celeba')
            if os.path.exists(torchvision_celeba) and torchvision_celeba != self.data_dir:
                import shutil
                if os.path.exists(self.data_dir):
                    shutil.rmtree(self.data_dir)
                shutil.move(torchvision_celeba, self.data_dir)
                print(f"✓ Moved files to {self.data_dir}")
            
            return True
            
        except ImportError:
            print("✗ torchvision not installed. Install with: pip install torchvision")
            return False
        except Exception as e:
            print(f"✗ Error using torchvision: {e}")
            return False
    
    def download_with_kaggle(self) -> bool:
        """
        Alternative: Download from Kaggle (requires Kaggle API setup).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import kaggle
            
            print("Attempting to download from Kaggle...")
            print("Note: This requires Kaggle API credentials.")
            print("Set up credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                'jessicali9530/celeba-dataset',
                path=self.data_dir,
                unzip=True
            )
            
            print("✓ Successfully downloaded from Kaggle")
            return True
            
        except ImportError:
            print("✗ kaggle package not installed.")
            print("  To use Kaggle download:")
            print("  1. pip install kaggle")
            print("  2. Set up API credentials")
            return False
        except Exception as e:
            print(f"✗ Error downloading from Kaggle: {e}")
            return False
    
    def manual_download_instructions(self):
        """Print detailed manual download instructions."""
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*70)
        print("\nOption 1: Download from official website")
        print("-" * 40)
        print("1. Visit: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("2. You may need to:")
        print("   - Create an account")
        print("   - Fill out a form for dataset access")
        print("3. Download these files:")
        print("   - img_align_celeba.zip (1.4GB)")
        print("   - list_eval_partition.txt")
        print(f"4. Place files in: {self.data_dir}")
        print("5. Run: python download_celeba.py --verify-only")
        
        print("\nOption 2: Use torchvision (automatic)")
        print("-" * 40)
        print("1. Install torchvision: pip install torchvision")
        print("2. Run: python download_celeba.py --use-torchvision")
        
        print("\nOption 3: Download from Kaggle")
        print("-" * 40)
        print("1. Create Kaggle account: https://www.kaggle.com")
        print("2. Get API credentials: https://www.kaggle.com/settings")
        print("3. Install: pip install kaggle")
        print("4. Run: python download_celeba.py --use-kaggle")
        
        print("\nOption 4: Download from alternative mirrors")
        print("-" * 40)
        print("Search for 'CelebA dataset mirror' or try:")
        print("- Academic Torrents")
        print("- Baidu Drive (for users in China)")
        print("- Contact the dataset authors directly")
        print("="*70)
    
    def download_all(self, include_extras: bool = False, remove_zips: bool = False, use_torchvision: bool = False, use_kaggle: bool = False) -> bool:
        """
        Download all CelebA files using the specified method.
        
        Args:
            include_extras: Whether to download attribute and landmark files
            remove_zips: Whether to remove zip files after extraction to save space
            use_torchvision: Use torchvision for downloading (more reliable)
            use_kaggle: Use Kaggle API for downloading
            
        Returns:
            True if all downloads successful, False otherwise
        """
        # Try alternative methods first if specified
        if use_torchvision:
            return self.download_with_torchvision()
        
        if use_kaggle:
            return self.download_with_kaggle()
        
        success = True
        
        # Essential files (images and partition)
        essential_files = ['img_align_celeba.zip', 'list_eval_partition.txt']
        
        # Extra annotation files
        extra_files = [
            'identity_CelebA.txt',
            'list_attr_celeba.txt',
            'list_bbox_celeba.txt',
            'list_landmarks_align_celeba.txt'
        ]
        
        files_to_download = essential_files
        if include_extras:
            files_to_download.extend(extra_files)
        
        if not self.quiet:
            print("=" * 60)
            print("CelebA Dataset Downloader")
            print("=" * 60)
            print(f"Download directory: {self.data_dir}")
            print(f"Files to download: {len(files_to_download)}")
            if 'img_align_celeba.zip' in files_to_download:
                print("Total download size: ~1.4GB")
            print("=" * 60)
            print()
        
        # Download files
        for i, file_name in enumerate(files_to_download, 1):
            if not self.quiet:
                print(f"[{i}/{len(files_to_download)}] Processing {file_name}")
            
            if not self.download_from_gdrive(file_name):
                print(f"✗ Failed to download {file_name}")
                success = False
            
            if not self.quiet:
                print()
        
        # Extract images
        if 'img_align_celeba.zip' in files_to_download and success:
            if not self.extract_zip('img_align_celeba.zip', remove_after=remove_zips):
                success = False
        
        return success
    
    def setup_splits(self) -> bool:
        """
        Create train/val/test split files based on official partition.
        
        Returns:
            True if successful, False otherwise
        """
        partition_file = os.path.join(self.data_dir, 'list_eval_partition.txt')
        
        if not os.path.exists(partition_file):
            print("✗ Partition file not found")
            print("  Please download it first with: python download_celeba.py")
            return False
        
        if not self.quiet:
            print("📝 Setting up official CelebA splits...")
        
        train_list = []
        val_list = []
        test_list = []
        
        try:
            with open(partition_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        filename, partition = parts
                        partition = int(partition)
                        
                        if partition == 0:
                            train_list.append(filename)
                        elif partition == 1:
                            val_list.append(filename)
                        elif partition == 2:
                            test_list.append(filename)
            
            # Save split lists
            splits = {
                'train.txt': train_list,
                'val.txt': val_list,
                'test.txt': test_list
            }
            
            for split_file, file_list in splits.items():
                split_path = os.path.join(self.data_dir, split_file)
                with open(split_path, 'w') as f:
                    for filename in file_list:
                        f.write(f"{filename}\n")
                if not self.quiet:
                    print(f"  Created {split_file}: {len(file_list):,} images")
            
            return True
            
        except Exception as e:
            print(f"✗ Error setting up splits: {e}")
            return False
    
    def verify_dataset(self) -> bool:
        """
        Verify that the dataset is properly set up.
        
        Returns:
            True if dataset is ready, False otherwise
        """
        img_dir = os.path.join(self.data_dir, 'img_align_celeba')
        
        if not os.path.exists(img_dir):
            print(f"✗ Image directory not found: {img_dir}")
            return False
        
        # Count images
        images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        num_images = len(images)
        expected_images = 202599
        
        if num_images != expected_images:
            print(f"✗ Image count mismatch:")
            print(f"  Expected: {expected_images:,}")
            print(f"  Found: {num_images:,}")
            return False
        
        # Check partition file
        partition_file = os.path.join(self.data_dir, 'list_eval_partition.txt')
        if not os.path.exists(partition_file):
            print(f"⚠ Warning: Partition file not found")
            print("  The dataset will use default 80/10/10 splits")
        
        if not self.quiet:
            print(f"✓ Dataset verified: {num_images:,} images ready for training!")
        
        return True
    
    def cleanup(self):
        """Remove downloaded zip files to save disk space."""
        zip_files = ['img_align_celeba.zip']
        
        for zip_file in zip_files:
            zip_path = os.path.join(self.data_dir, zip_file)
            if os.path.exists(zip_path):
                os.remove(zip_path)
                if not self.quiet:
                    print(f"Removed {zip_file} to save disk space")
    
    def get_dataset_info(self) -> Dict:
        """Get information about the downloaded dataset."""
        info = {
            'data_dir': self.data_dir,
            'images_found': 0,
            'has_partition_file': False,
            'has_attributes': False,
            'has_landmarks': False,
            'total_size_mb': 0
        }
        
        # Check images
        img_dir = os.path.join(self.data_dir, 'img_align_celeba')
        if os.path.exists(img_dir):
            info['images_found'] = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # Check other files
        info['has_partition_file'] = os.path.exists(os.path.join(self.data_dir, 'list_eval_partition.txt'))
        info['has_attributes'] = os.path.exists(os.path.join(self.data_dir, 'list_attr_celeba.txt'))
        info['has_landmarks'] = os.path.exists(os.path.join(self.data_dir, 'list_landmarks_align_celeba.txt'))
        
        # Calculate total size
        total_size = 0
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        info['total_size_mb'] = total_size / (1024 * 1024)
        
        return info


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare CelebA dataset for VAE inpainting training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Try to download from Google Drive (may fail due to quotas)
  python download_celeba.py
  
  # Use torchvision to download (RECOMMENDED - more reliable)
  python download_celeba.py --use-torchvision
  
  # Use Kaggle API to download (requires Kaggle account)
  python download_celeba.py --use-kaggle
  
  # Download all files including attributes and landmarks
  python download_celeba.py --include-extras
  
  # Download and remove zips after extraction (saves ~1.4GB)
  python download_celeba.py --remove-zips
  
  # Verify existing dataset
  python download_celeba.py --verify-only
  
  # Get dataset information
  python download_celeba.py --info
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='./assets/datasets/celeba',
                        help='Directory to save the dataset (default: ./assets/datasets/celeba)')
    parser.add_argument('--include-extras', action='store_true',
                        help='Download attribute and landmark files (not required for basic inpainting)')
    parser.add_argument('--remove-zips', action='store_true',
                        help='Remove zip files after extraction to save disk space (~1.4GB)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing dataset without downloading')
    parser.add_argument('--info', action='store_true',
                        help='Show information about the dataset')
    parser.add_argument('--use-torchvision', action='store_true',
                        help='Use torchvision to download (RECOMMENDED - more reliable)')
    parser.add_argument('--use-kaggle', action='store_true',
                        help='Use Kaggle API to download (requires Kaggle credentials)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if files exist')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = CelebADownloader(args.data_dir, quiet=args.quiet)
    
    # Handle different modes
    if args.info:
        info = downloader.get_dataset_info()
        print("\nCelebA Dataset Information")
        print("=" * 40)
        print(f"Directory: {info['data_dir']}")
        print(f"Images found: {info['images_found']:,}/202,599")
        print(f"Has partition file: {'Yes' if info['has_partition_file'] else 'No'}")
        print(f"Has attributes: {'Yes' if info['has_attributes'] else 'No'}")
        print(f"Has landmarks: {'Yes' if info['has_landmarks'] else 'No'}")
        print(f"Total size: {info['total_size_mb']:.1f} MB")
        print("=" * 40)
        
    elif args.verify_only:
        if downloader.verify_dataset():
            print("\n✅ Dataset is ready for training!")
        else:
            print("\n❌ Dataset verification failed")
            print("\nTo download the dataset, run:")
            print("  python download_celeba.py --use-torchvision")
            sys.exit(1)
    
    else:
        # Download mode
        print("\n🚀 Starting CelebA dataset download...")
        
        # Check if should use alternative download methods
        if args.use_torchvision:
            print("Using torchvision method (most reliable)...")
            if downloader.download_with_torchvision():
                print("\n✅ Dataset downloaded successfully with torchvision!")
                sys.exit(0)
            else:
                print("\n❌ Torchvision download failed")
                downloader.manual_download_instructions()
                sys.exit(1)
        
        elif args.use_kaggle:
            print("Using Kaggle API method...")
            if downloader.download_with_kaggle():
                print("\n✅ Dataset downloaded successfully from Kaggle!")
                sys.exit(0)
            else:
                print("\n❌ Kaggle download failed")
                downloader.manual_download_instructions()
                sys.exit(1)
        
        # Default: try Google Drive (may fail due to quotas)
        print("Attempting Google Drive download (may fail due to quotas)...")
        print("This will download ~1.4GB of data.\n")
        
        # Check available disk space
        import shutil
        free_space = shutil.disk_usage(args.data_dir).free / (1024**3)  # GB
        required_space = 3.0 if not args.remove_zips else 1.5  # GB
        
        if free_space < required_space:
            print(f"⚠ Warning: Low disk space!")
            print(f"  Available: {free_space:.1f} GB")
            print(f"  Required: {required_space:.1f} GB")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        
        # Download all files
        if downloader.download_all(
            include_extras=args.include_extras,
            remove_zips=args.remove_zips,
            use_torchvision=args.use_torchvision,
            use_kaggle=args.use_kaggle
        ):
            print("\n✅ All downloads completed successfully!")
            
            # Set up splits
            downloader.setup_splits()
            
            # Final verification
            if downloader.verify_dataset():
                print("\n🎉 Dataset is ready for training!")
                print(f"\nYou can now start training with:")
                print(f"  python scripts/train.py --config config/default.yaml")
            else:
                print("\n⚠ Dataset downloaded but verification failed")
        else:
            print("\n❌ Google Drive download failed (common issue)")
            print("\n" + "="*70)
            print("RECOMMENDED SOLUTION: Use torchvision instead")
            print("="*70)
            print("\nThe Google Drive links often have quota issues.")
            print("Try this more reliable method instead:\n")
            print("  python download_celeba.py --use-torchvision\n")
            print("This uses PyTorch's official mirrors which are more reliable.")
            print("="*70)
            
            downloader.manual_download_instructions()
            sys.exit(1)


if __name__ == '__main__':
    main()