#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cleanup Script for Stock Market Agent.

This script recursively deletes all files and directories in the data/, models/,
and outputs/ folders, while preserving README.md files in each folder.

Usage:
    python cleanup.py              # Dry run (shows what would be deleted)
    python cleanup.py --confirm    # Actually delete files
    python cleanup.py --help       # Show help message

Safety Features:
    - Dry run by default (requires --confirm to actually delete)
    - Preserves README.md files (case-insensitive)
    - Shows summary of files/folders to be deleted before confirmation
    - Handles permission errors gracefully
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse  # Parse command-line arguments.
import os  # File and directory operations.
import shutil  # Remove directory trees.
import sys  # Exit codes and stderr.
from pathlib import Path  # Cross-platform path handling.
from typing import List, Tuple  # Type hints for function signatures.


# =============================================================================
# CONFIGURATION
# =============================================================================

# Directories to clean (relative to script location).
TARGET_DIRS = ["data", "models", "outputs"]  # Folders to clean up.

# Files to preserve (case-insensitive matching).
PRESERVE_FILES = ["readme.md", "readme.txt", "readme"]  # README files to keep.


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def eprint(msg: str) -> None:
    """
    Print error message to stderr.
    
    Args:
        msg: The error message to print.
    """
    print(msg, file=sys.stderr)  # Print to stderr for error visibility.


def should_preserve(path: Path) -> bool:
    """
    Check if a file should be preserved (not deleted).
    
    Args:
        path: Path to the file to check.
        
    Returns:
        True if the file should be preserved, False otherwise.
    """
    # Check if filename (case-insensitive) matches any preserve pattern.
    return path.name.lower() in PRESERVE_FILES  # Case-insensitive comparison.


def get_items_to_delete(base_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Get lists of files and directories to delete in a target directory.
    
    Recursively scans the directory and identifies all items that should be
    deleted, excluding preserved files (README.md).
    
    Args:
        base_dir: The base directory to scan.
        
    Returns:
        Tuple of (files_to_delete, dirs_to_delete).
        Directories are sorted by depth (deepest first) for safe deletion.
    """
    files_to_delete: List[Path] = []  # List of files to delete.
    dirs_to_delete: List[Path] = []  # List of directories to delete.
    
    if not base_dir.exists():  # Skip if directory doesn't exist.
        return files_to_delete, dirs_to_delete
    
    # Walk the directory tree.
    for root, dirs, files in os.walk(base_dir, topdown=False):  # Bottom-up traversal.
        root_path = Path(root)  # Convert to Path object.
        
        # Collect files to delete (excluding preserved files).
        for file_name in files:  # Iterate over files in current directory.
            file_path = root_path / file_name  # Full path to file.
            if not should_preserve(file_path):  # Check if file should be deleted.
                files_to_delete.append(file_path)  # Add to deletion list.
        
        # Collect directories to delete (will check if empty after file deletion).
        if root_path != base_dir:  # Don't delete the base directory itself.
            dirs_to_delete.append(root_path)  # Add to deletion list.
    
    # Sort directories by depth (deepest first) for safe deletion order.
    dirs_to_delete.sort(key=lambda p: len(p.parts), reverse=True)
    
    return files_to_delete, dirs_to_delete


def delete_file(file_path: Path, dry_run: bool = True) -> bool:
    """
    Delete a single file.
    
    Args:
        file_path: Path to the file to delete.
        dry_run: If True, only print what would be deleted.
        
    Returns:
        True if deletion was successful (or dry run), False on error.
    """
    try:
        if dry_run:  # Dry run mode - just print.
            print(f"  [DRY RUN] Would delete file: {file_path}")
            return True
        else:  # Actually delete the file.
            file_path.unlink()  # Delete the file.
            print(f"  ✓ Deleted file: {file_path}")
            return True
    except PermissionError:  # Handle permission errors.
        eprint(f"  ✗ Permission denied: {file_path}")
        return False
    except OSError as e:  # Handle other OS errors.
        eprint(f"  ✗ Error deleting {file_path}: {e}")
        return False


def delete_directory(dir_path: Path, dry_run: bool = True) -> bool:
    """
    Delete a directory if it's empty or contains only deletable items.
    
    Args:
        dir_path: Path to the directory to delete.
        dry_run: If True, only print what would be deleted.
        
    Returns:
        True if deletion was successful (or dry run), False on error.
    """
    try:
        # Check if directory is empty or only contains preserved files.
        remaining_items = list(dir_path.iterdir())  # Get remaining items.
        
        # Check if all remaining items are preserved files.
        all_preserved = all(should_preserve(item) for item in remaining_items if item.is_file())
        has_subdirs = any(item.is_dir() for item in remaining_items)
        
        if remaining_items and (has_subdirs or not all_preserved):
            # Directory still has non-preserved content, skip.
            return True  # Not an error, just can't delete yet.
        
        if dry_run:  # Dry run mode - just print.
            if not remaining_items:  # Only report if truly empty.
                print(f"  [DRY RUN] Would delete empty dir: {dir_path}")
            return True
        else:  # Actually delete the directory.
            if not remaining_items:  # Only delete if empty.
                dir_path.rmdir()  # Delete empty directory.
                print(f"  ✓ Deleted empty dir: {dir_path}")
            return True
    except PermissionError:  # Handle permission errors.
        eprint(f"  ✗ Permission denied: {dir_path}")
        return False
    except OSError as e:  # Handle other OS errors.
        eprint(f"  ✗ Error deleting {dir_path}: {e}")
        return False


def cleanup_directory(base_dir: Path, dry_run: bool = True) -> Tuple[int, int, int]:
    """
    Clean up a single target directory.
    
    Args:
        base_dir: The directory to clean.
        dry_run: If True, only print what would be deleted.
        
    Returns:
        Tuple of (files_deleted, dirs_deleted, errors).
    """
    print(f"\n{'='*60}")
    print(f"Cleaning: {base_dir}")
    print(f"{'='*60}")
    
    if not base_dir.exists():  # Check if directory exists.
        print(f"  Directory does not exist, skipping.")
        return 0, 0, 0
    
    # Get items to delete.
    files_to_delete, dirs_to_delete = get_items_to_delete(base_dir)
    
    if not files_to_delete and not dirs_to_delete:  # Nothing to delete.
        print(f"  No files or directories to delete.")
        return 0, 0, 0
    
    # Track statistics.
    files_deleted = 0  # Count of deleted files.
    dirs_deleted = 0  # Count of deleted directories.
    errors = 0  # Count of errors.
    
    # Delete files first.
    if files_to_delete:  # Check if there are files to delete.
        print(f"\n  Files to delete: {len(files_to_delete)}")
        for file_path in files_to_delete:  # Iterate over files.
            if delete_file(file_path, dry_run):  # Attempt deletion.
                files_deleted += 1  # Increment counter.
            else:
                errors += 1  # Increment error counter.
    
    # Then delete empty directories (deepest first).
    if dirs_to_delete:  # Check if there are directories to delete.
        print(f"\n  Directories to check: {len(dirs_to_delete)}")
        for dir_path in dirs_to_delete:  # Iterate over directories.
            if dir_path.exists():  # Check if still exists.
                if delete_directory(dir_path, dry_run):  # Attempt deletion.
                    dirs_deleted += 1  # Increment counter.
                else:
                    errors += 1  # Increment error counter.
    
    return files_deleted, dirs_deleted, errors


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> int:
    """
    Main entry point for the cleanup script.
    
    Returns:
        Exit code (0 for success, 1 for errors).
    """
    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(
        description="Clean up data, models, and outputs directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cleanup.py              # Dry run - shows what would be deleted
    python cleanup.py --confirm    # Actually delete files
    python cleanup.py --verbose    # Show preserved files too

Notes:
    - README.md files are always preserved
    - Directories are only deleted when empty
    - Use --confirm to actually delete files (dry run by default)
        """
    )
    
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete files (without this flag, only shows what would be deleted)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show preserved files in output"
    )
    
    args = parser.parse_args()  # Parse command-line arguments.
    
    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    
    # Get the project root directory (where this script is located).
    script_dir = Path(__file__).resolve().parent  # Script's directory.
    
    # Determine if this is a dry run.
    dry_run = not args.confirm  # Dry run unless --confirm is specified.
    
    # -------------------------------------------------------------------------
    # PRINT HEADER
    # -------------------------------------------------------------------------
    
    print("=" * 60)
    print("STOCK MARKET AGENT - CLEANUP SCRIPT")
    print("=" * 60)
    print(f"Project directory: {script_dir}")
    print(f"Target directories: {', '.join(TARGET_DIRS)}")
    print(f"Preserved files: {', '.join(PRESERVE_FILES)}")
    print(f"Mode: {'DRY RUN (use --confirm to delete)' if dry_run else 'DELETE MODE'}")
    
    # -------------------------------------------------------------------------
    # CLEANUP EACH TARGET DIRECTORY
    # -------------------------------------------------------------------------
    
    total_files = 0  # Total files deleted.
    total_dirs = 0  # Total directories deleted.
    total_errors = 0  # Total errors encountered.
    
    for dir_name in TARGET_DIRS:  # Iterate over target directories.
        target_dir = script_dir / dir_name  # Full path to target directory.
        files, dirs, errors = cleanup_directory(target_dir, dry_run)  # Clean up.
        total_files += files  # Accumulate file count.
        total_dirs += dirs  # Accumulate directory count.
        total_errors += errors  # Accumulate error count.
    
    # -------------------------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if dry_run:  # Dry run summary.
        print(f"Files that would be deleted: {total_files}")
        print(f"Directories that would be deleted: {total_dirs}")
        if total_files > 0 or total_dirs > 0:
            print(f"\n⚠️  Run with --confirm to actually delete these items.")
    else:  # Actual deletion summary.
        print(f"Files deleted: {total_files}")
        print(f"Directories deleted: {total_dirs}")
        print(f"Errors: {total_errors}")
        if total_errors == 0:
            print(f"\n✅ Cleanup completed successfully!")
        else:
            print(f"\n⚠️  Cleanup completed with {total_errors} error(s).")
    
    # Return exit code.
    return 1 if total_errors > 0 else 0


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())  # Run main function and exit with its return code.
