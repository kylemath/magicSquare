import numpy as np

def get_magic_squares_4x4():
    """Return three different 4x4 magic squares"""
    return [
        # Classic magic square (sum = 34)
        np.array([
            [16, 3, 2, 13],
            [5, 10, 11, 8],
            [9, 6, 7, 12],
            [4, 15, 14, 1]
        ]),
        
        # Dürer's magic square (sum = 34)
        np.array([
            [16, 3, 2, 13],
            [5, 10, 11, 8],
            [9, 6, 7, 12],
            [4, 15, 14, 1]
        ]),
        
        # Alternate magic square (sum = 34)
        np.array([
            [1, 15, 14, 4],
            [12, 6, 7, 9],
            [8, 10, 11, 5],
            [13, 3, 2, 16]
        ])
    ]

def map_to_circle(square, pattern="clockwise"):
    """Map a magic square to circular arrangement using different patterns"""
    if pattern == "clockwise":
        read_order = [
            (0,0), (0,1), (0,2), (0,3),  # Top row
            (1,3), (2,3), (3,3),         # Right column
            (3,2), (3,1), (3,0),         # Bottom row
            (2,0), (1,0),                # Left column
            (1,1), (1,2), (2,2), (2,1)   # Inner square clockwise
        ]
    elif pattern == "spiral":
        read_order = [
            (0,0), (0,1), (0,2), (0,3),
            (1,3), (2,3), (3,3), (3,2),
            (3,1), (3,0), (2,0), (1,0),
            (1,1), (1,2), (2,2), (2,1)
        ]
    elif pattern == "alternating":
        read_order = [
            (0,0), (0,2), (0,1), (0,3),  # Top alternating
            (1,3), (3,3), (2,3),         # Right alternating
            (3,2), (3,0), (3,1),         # Bottom alternating
            (2,0), (1,0),                # Left alternating
            (1,1), (2,2), (1,2), (2,1)   # Inner alternating
        ]
    
    return np.array([square[i,j] for i,j in read_order])

def evaluate_balance(weights):
    """Enhanced balance evaluation"""
    n = len(weights)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Static balance (center of mass)
    com_x = sum(w * np.cos(t) for w, t in zip(weights, theta))
    com_y = sum(w * np.sin(t) for w, t in zip(weights, theta))
    static_imbalance = np.sqrt(com_x**2 + com_y**2)
    
    # Dynamic balance (improved)
    I_xx = sum(w * (np.sin(t))**2 for w, t in zip(weights, theta))
    I_yy = sum(w * (np.cos(t))**2 for w, t in zip(weights, theta))
    I_xy = sum(w * np.cos(t) * np.sin(t) for w, t in zip(weights, theta))
    
    # Principal moments of inertia
    I_avg = (I_xx + I_yy) / 2
    I_diff = np.sqrt((I_xx - I_yy)**2 + 4*I_xy**2)
    I_max = I_avg + I_diff/2
    I_min = I_avg - I_diff/2
    
    # Dynamic imbalance considering ratio of principal moments
    dynamic_imbalance = (I_max - I_min) / I_avg
    
    # Weight the imbalances based on angular velocity
    omega = angular_velocity  # You'd need to pass this in
    static_weight = 1.0 / (1.0 + (omega/CRITICAL_SPEED)**2)
    dynamic_weight = 1.0 - static_weight
    
    total_imbalance = static_weight * static_imbalance + dynamic_weight * dynamic_imbalance
    
    return total_imbalance, static_imbalance, dynamic_imbalance

# Our best solution from optimization
optimized_weights = np.array([7, 4, 10, 12, 6, 13, 1, 14, 2, 15, 11, 3, 9, 5, 16, 8])
static_opt, dynamic_opt = evaluate_balance(optimized_weights)
print("\nOptimized solution:")
print(f"Weights: {optimized_weights}")
print(f"Static imbalance: {static_opt:.6f}")
print(f"Dynamic imbalance: {dynamic_opt:.6f}")
print(f"Total imbalance: {static_opt + dynamic_opt:.6f}")

# Test magic square mappings
squares = get_magic_squares_4x4()
patterns = ["clockwise", "spiral", "alternating"]

print("\nMagic square mappings:")
for i, square in enumerate(squares):
    print(f"\nMagic Square {i+1}:")
    for pattern in patterns:
        circular = map_to_circle(square, pattern)
        static, dynamic = evaluate_balance(circular)
        print(f"\n{pattern.capitalize()} pattern:")
        print(f"Weights: {circular}")
        print(f"Static imbalance: {static:.6f}")
        print(f"Dynamic imbalance: {dynamic:.6f}")
        print(f"Total imbalance: {static + dynamic:.6f}") 