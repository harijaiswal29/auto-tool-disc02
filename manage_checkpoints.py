#!/usr/bin/env python3
"""
Checkpoint Management Utility
=============================
Helps manage and organize checkpoint files from evaluation runs.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import argparse

def list_checkpoint_directories():
    """List all checkpoint directories and their contents"""
    results_dir = Path("tests/dissertation_test_suite/results")
    
    print("\n" + "="*70)
    print("CHECKPOINT DIRECTORIES")
    print("="*70)
    
    # Find all directories with checkpoints
    checkpoint_dirs = []
    for result_dir in results_dir.glob("dissert-result-*"):
        checkpoint_dir = result_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.pkl"))
            if checkpoint_files:
                checkpoint_dirs.append({
                    "path": checkpoint_dir,
                    "parent": result_dir,
                    "files": checkpoint_files,
                    "count": len(checkpoint_files),
                    "size": sum(f.stat().st_size for f in checkpoint_files)
                })
    
    if not checkpoint_dirs:
        print("No checkpoint directories found.")
        return []
    
    # Sort by modification time
    checkpoint_dirs.sort(key=lambda x: x["parent"].stat().st_mtime, reverse=True)
    
    for i, cp_dir in enumerate(checkpoint_dirs):
        mtime = datetime.fromtimestamp(cp_dir["parent"].stat().st_mtime)
        size_mb = cp_dir["size"] / (1024 * 1024)
        
        print(f"\n{i+1}. {cp_dir['parent'].name}")
        print(f"   Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Checkpoints: {cp_dir['count']} files")
        print(f"   Total size: {size_mb:.2f} MB")
        print(f"   Path: {cp_dir['path']}")
        
        # Show sample checkpoint files
        if cp_dir['files']:
            print("   Sample files:")
            for f in sorted(cp_dir['files'])[:3]:
                print(f"     - {f.name}")
    
    return checkpoint_dirs

def archive_old_checkpoints(keep_latest=2):
    """Archive old checkpoint directories, keeping only the latest N"""
    checkpoint_dirs = list_checkpoint_directories()
    
    if len(checkpoint_dirs) <= keep_latest:
        print(f"\n✅ Only {len(checkpoint_dirs)} checkpoint directories found. Nothing to archive.")
        return
    
    print(f"\n" + "="*70)
    print(f"ARCHIVING OLD CHECKPOINTS (keeping latest {keep_latest})")
    print("="*70)
    
    # Create archive directory
    archive_dir = Path("tests/dissertation_test_suite/results/archived_checkpoints")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Archive older directories
    dirs_to_archive = checkpoint_dirs[keep_latest:]
    
    for cp_dir in dirs_to_archive:
        archive_name = f"{cp_dir['parent'].name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_path = archive_dir / archive_name
        
        print(f"\nArchiving: {cp_dir['parent'].name}")
        print(f"  From: {cp_dir['parent']}")
        print(f"  To: {archive_path}")
        
        try:
            shutil.move(str(cp_dir['parent']), str(archive_path))
            print(f"  ✅ Archived successfully")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Archived {len(dirs_to_archive)} old checkpoint directories")
    print(f"   Archive location: {archive_dir}")

def clean_duplicate_checkpoints():
    """Remove duplicate checkpoint files within each directory"""
    checkpoint_dirs = list_checkpoint_directories()
    
    print(f"\n" + "="*70)
    print("CLEANING DUPLICATE CHECKPOINTS")
    print("="*70)
    
    total_removed = 0
    total_saved = 0
    
    for cp_dir in checkpoint_dirs:
        print(f"\nCleaning: {cp_dir['parent'].name}")
        
        # Group checkpoints by strategy and episode
        checkpoint_groups = {}
        for ckpt_file in cp_dir['files']:
            # Parse filename: checkpoint_{strategy}_ep{episode}_{timestamp}.pkl
            parts = ckpt_file.stem.split('_')
            if len(parts) >= 4 and parts[0] == 'checkpoint':
                strategy = '_'.join(parts[1:-2])  # Handle multi-word strategies
                episode = parts[-2]  # ep{number}
                key = f"{strategy}_{episode}"
                
                if key not in checkpoint_groups:
                    checkpoint_groups[key] = []
                checkpoint_groups[key].append(ckpt_file)
        
        # Keep only the latest checkpoint for each strategy-episode combination
        removed = 0
        saved_bytes = 0
        for key, files in checkpoint_groups.items():
            if len(files) > 1:
                # Sort by modification time, keep the latest
                files.sort(key=lambda f: f.stat().st_mtime)
                for old_file in files[:-1]:
                    saved_bytes += old_file.stat().st_size
                    print(f"  Removing duplicate: {old_file.name}")
                    old_file.unlink()
                    removed += 1
        
        if removed > 0:
            print(f"  ✅ Removed {removed} duplicates, saved {saved_bytes/(1024*1024):.2f} MB")
        else:
            print(f"  ✅ No duplicates found")
        
        total_removed += removed
        total_saved += saved_bytes
    
    print(f"\n✅ Total: Removed {total_removed} duplicate files, saved {total_saved/(1024*1024):.2f} MB")

def get_checkpoint_info(checkpoint_path):
    """Get information about a specific checkpoint directory"""
    path = Path(checkpoint_path)
    
    if not path.exists():
        print(f"❌ Directory not found: {path}")
        return
    
    print(f"\n" + "="*70)
    print(f"CHECKPOINT INFO: {path.name}")
    print("="*70)
    
    # Check for run config
    config_file = path.parent / "run_config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print("\nRun Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # List checkpoint files
    checkpoint_files = list(path.glob("*.pkl"))
    print(f"\nCheckpoint Files: {len(checkpoint_files)}")
    
    # Group by strategy
    strategies = {}
    for f in checkpoint_files:
        parts = f.stem.split('_')
        if len(parts) >= 4:
            strategy = '_'.join(parts[1:-2])
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(f)
    
    for strategy, files in strategies.items():
        episodes = []
        for f in files:
            if 'ep' in f.stem:
                ep_num = f.stem.split('ep')[-1].split('_')[0]
                try:
                    episodes.append(int(ep_num))
                except:
                    pass
        
        if episodes:
            print(f"\n  {strategy}:")
            print(f"    Files: {len(files)}")
            print(f"    Episodes: {min(episodes)}-{max(episodes)}")
            print(f"    Checkpoints at: {sorted(episodes)}")

def main():
    parser = argparse.ArgumentParser(description="Manage evaluation checkpoints")
    parser.add_argument('action', choices=['list', 'archive', 'clean', 'info'],
                       help='Action to perform')
    parser.add_argument('--keep', type=int, default=2,
                       help='Number of latest runs to keep when archiving')
    parser.add_argument('--path', type=str,
                       help='Path to checkpoint directory for info action')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_checkpoint_directories()
    elif args.action == 'archive':
        archive_old_checkpoints(args.keep)
    elif args.action == 'clean':
        clean_duplicate_checkpoints()
    elif args.action == 'info':
        if args.path:
            get_checkpoint_info(args.path)
        else:
            print("❌ Please provide --path for info action")

if __name__ == "__main__":
    main()