import numpy as np
import glob
import os

def load_latest_squares():
    """Load the most recently saved magic squares file"""
    # Find all .npy files with magic squares
    files = glob.glob("magic_squares_*.npy")
    if not files:
        print("No magic squares files found!")
        return None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading squares from: {latest_file}")
    
    # Load the squares
    squares = np.load(latest_file)
    print(f"Loaded {len(squares)} squares")
    
    return squares

def get_all_transformations(square):
    """Get all possible rotations and reflections of a square"""
    transformations = []
    
    # Get all 4 rotations
    current = square.copy()
    for _ in range(4):
        transformations.append(current.copy())
        current = np.rot90(current)
    
    # Get all 4 rotations of the reflection
    current = np.fliplr(square).copy()
    for _ in range(4):
        transformations.append(current.copy())
        current = np.rot90(current)
    
    return transformations

def find_unique_squares(squares):
    """Find unique squares ignoring rotations and reflections"""
    unique_squares = []
    seen = set()
    
    for square in squares:
        # Get all possible transformations of this square
        transforms = get_all_transformations(square)
        
        # Check if we've seen any of these transformations
        found = False
        for transform in transforms:
            if tuple(map(tuple, transform)) in seen:
                found = True
                break
        
        # If we haven't seen this square or any of its transformations, add it
        if not found:
            unique_squares.append(square)
            # Add all transformations to seen set
            for transform in transforms:
                seen.add(tuple(map(tuple, transform)))
    
    return unique_squares

def remove_exact_duplicates(squares):
    """Remove exact duplicate squares (identical arrays)"""
    unique_squares = []
    seen = set()
    
    print("\nRemoving exact duplicates:")
    print(f"Starting with {len(squares)} squares")
    
    for square in squares:
        # Convert to tuple for hashability
        square_tuple = tuple(map(tuple, square))
        if square_tuple not in seen:
            seen.add(square_tuple)
            unique_squares.append(square)
    
    print(f"Found {len(squares) - len(unique_squares)} exact duplicates")
    print(f"Remaining squares: {len(unique_squares)}")
    
    return np.array(unique_squares)

def validate_magic_square(square):
    """Comprehensive validation of a magic square"""
    n = 4  # We're working with 4x4 squares
    magic_sum = 34
    
    # Check all numbers 1-16 are used exactly once
    numbers = square.flatten()
    if not all(n in numbers for n in range(1, 17)):
        return False, "Not all numbers 1-16 are used"
    
    # Check rows, columns, and diagonals sum to 34
    for i in range(n):
        if sum(square[i]) != magic_sum:
            return False, f"Row {i} sum is {sum(square[i])}"
        if sum(square[:,i]) != magic_sum:
            return False, f"Column {i} sum is {sum(square[:,i])}"
    
    if sum(square.diagonal()) != magic_sum:
        return False, "Main diagonal sum is wrong"
    if sum(np.fliplr(square).diagonal()) != magic_sum:
        return False, "Other diagonal sum is wrong"
    
    return True, "Valid magic square"

def normalize_square(square):
    """Normalize square orientation to have smallest corner in top-left,
    then smallest possible value in position (0,1)"""
    n = 4
    
    # Get corners of original square
    corners = [square[0,0], square[0,3], square[3,0], square[3,3]]
    min_corner = min(corners)
    
    # Get all transformations
    transforms = get_all_transformations(square)
    
    # Find transformations with min_corner in top-left
    valid_transforms = []
    for t in transforms:
        if t[0,0] == min_corner:
            valid_transforms.append(t)
    
    if not valid_transforms:
        print(f"Warning: Could not find transformation with {min_corner} in top-left")
        print(f"Original square:\n{square}")
        return square
    
    # Among valid transforms, find one with smallest value in position (0,1)
    best_transform = min(valid_transforms, key=lambda t: t[0,1])
    
    return best_transform

def validate_normalization(squares):
    """Validate that all squares are properly normalized"""
    print("\nValidating square normalization:")
    
    normalized_squares = []
    issues = []
    
    for i, square in enumerate(squares):
        norm_square = normalize_square(square)
        normalized_squares.append(norm_square)
        
        # Get corners of normalized square
        corners = [norm_square[0,0], norm_square[0,3], 
                  norm_square[3,0], norm_square[3,3]]
        min_corner = min(corners)
        
        # Check 1: Is the smallest corner in top-left?
        if norm_square[0,0] != min_corner:
            issues.append(f"Square {i}: Top-left {norm_square[0,0]} is not minimum corner value {min_corner}")
            continue
        
        # Check 2: Among squares with same top-left, is this the one with smallest (0,1)?
        transforms = [t for t in get_all_transformations(square) if t[0,0] == min_corner]
        min_second = min(t[0,1] for t in transforms)
        if norm_square[0,1] > min_second:
            issues.append(f"Square {i}: Position (0,1) value {norm_square[0,1]} is not minimum possible {min_second}")
            continue
    
    # Report results
    print(f"Normalized {len(squares)} squares")
    if not issues:
        print("✓ All squares properly normalized")
        print(f"  All squares have minimum corner value in top-left")
        # Show some statistics
        first_vals = [s[0,0] for s in normalized_squares]
        second_vals = [s[0,1] for s in normalized_squares]
        print(f"  Top-left values range: {min(first_vals)}-{max(first_vals)}")
        print(f"  Position (0,1) values range: {min(second_vals)}-{max(second_vals)}")
    else:
        print(f"! Found {len(issues)} normalization issues:")
        for issue in issues[:5]:
            print(f"  {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more issues")
    
    return normalized_squares

def reduce_to_unique(squares):
    """Remove duplicates and return 880 unique squares"""
    # First remove exact duplicates
    squares = remove_exact_duplicates(squares)
    # Then remove rotations/reflections
    unique_squares = find_unique_squares(squares)
    
    print("\nAnalyzing Unique Squares (ignoring rotations/reflections):")
    print(f"Found {len(unique_squares)} unique squares")
    print(f"Reduction ratio: {len(squares)}/{len(unique_squares)} = {len(squares)/len(unique_squares):.2f}")
    
    if len(unique_squares) == 880:
        print("✓ Successfully found all 880 known unique 4x4 magic squares!")
    else:
        print(f"! Warning: Expected 880 unique squares, found {len(unique_squares)}")
    
    return unique_squares