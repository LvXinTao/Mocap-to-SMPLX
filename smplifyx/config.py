OPTIMIZE_SHAPE={
    'shape3d':1.0,
    'reg_shape':5e-3
}

OPTIMIZE_RT={
    'k3d':1.0,
    'smooth_body':5e-1,
    'smooth_pose':1e-1,
    'reg_pose':1e-2
}

OPTIMIZE_POSES={
    'k3d':1,
    'smooth_body':5e-1,
    'smooth_pose':1e-1,
    'reg_pose':1e-2
}

OPTIMIZE_HAND={
    'k3d':1,
    'smooth_body':5,
    'smooth_pose':1e-1,
    'smooth_hand':1e-3,
    'reg_pose':1e-2,
    'k3d_hand':10, # 10
    'reg_hand':1e-4

}

OPTIMIZE_EXPR={
    'k3d':1,
    'smooth_body':5e-1,
    'smooth_pose':1e-1,
    'smooth_hand':1e-3,
    'smooth_head':1e-3,
    'reg_pose':1e-2,
    'k3d_hand':10,
    'reg_hand':1e-4,
    'k3d_face':10,
    'reg_head':1e-2,
    'reg_expr':1e-2

}