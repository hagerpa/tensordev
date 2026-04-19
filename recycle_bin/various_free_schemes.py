def cell_step(carry_cell, data_cell):
    u_w, f_w, g_w = carry_cell
    dy_j, u_sj, u_swj, f_sj, f_swj, g_sj, g_swj = data_cell

    def t_lin2(a, A, b, B):
        return t_sum(t_scal(A, a), t_scal(B, b))

    if _FREE_VOC_ORDER == -3:
        G_ij = t_inner(dx_i[:P], dy_j[:P])
        dx_adj_dy = adj_right(dx_i, dy_j, trunc=n)
        dy_adj_dx = adj_right(dy_j, dx_i, trunc=m)

        def to_dense_first_on(A):
            return (jnp.zeros_like(A[0][..., :1]),) + tuple(A)

        def forcing(u_val, f_val, g_val):
            return (
                    u_val * G_ij
                    + t_inner(f_val, dx_adj_dy)
                    + t_inner(g_val, dy_adj_dx)
            )

        def f_increment(u_val, f_val, g_val):
            return t_sum(
                t_sum(
                    t_scal(dx_i[:P_f], u_val),
                    t_prod(f_val, dx_i[:P_f], trunc=n),
                ),
                adj_left(g_val, dx_i, trunc=n),
            )

        def g_increment(u_val, f_val, g_val):
            return t_sum(
                t_sum(
                    t_scal(dy_j[:P_g], u_val),
                    t_prod(g_val, dy_j[:P_g], trunc=m),
                ),
                adj_left(f_val, dy_j, trunc=m),
            )

        # ----- stage 0: same one-sided predictor as in -1
        df0 = f_increment(u_sj, f_sj, g_sj)
        dg0 = g_increment(u_w, f_w, g_w)

        f_p = t_sum(f_sj, df0)
        g_p = t_sum(g_w, dg0)

        F_sw = forcing(u_swj, f_swj, g_swj)
        F_s = forcing(u_sj, f_sj, g_sj)
        F_w = forcing(u_w, f_w, g_w)

        u_base = u_sj + u_w - u_swj
        u_p = u_base + F_sw

        # ----- exact-VOC corner traces from affine edge data
        delta_x_u = u_w - u_swj
        delta_y_u = u_sj - u_swj

        delta_x_g = t_sub(g_w, g_swj)
        delta_y_f = t_sub(f_sj, f_swj)

        S_x = t_sum(
            t_scal(dx_i[:P_f], u_sj),
            adj_left(g_swj, dx_i, trunc=n),
        )
        T_x = t_sum(
            t_scal(dx_i[:P_f], delta_x_u),
            adj_left(delta_x_g, dx_i, trunc=n),
        )

        S_y = t_sum(
            t_scal(dy_j[:P_g], u_w),
            adj_left(f_swj, dy_j, trunc=m),
        )
        T_y = t_sum(
            t_scal(dy_j[:P_g], delta_y_u),
            adj_left(delta_y_f, dy_j, trunc=m),
        )

        f_voc = t_sum(
            core.tensor_fmexp(
                to_dense_first_on(f_sj),
                dx_i[:P_f],
                trunc=n,
                output_zero_level=False,
                k=0,
            ),
            t_sum(
                core.tensor_fmexp(
                    to_dense_first_on(S_x),
                    dx_i[:P_f],
                    trunc=n,
                    output_zero_level=False,
                    k=1,
                ),
                core.tensor_fmexp(
                    to_dense_first_on(T_x),
                    dx_i[:P_f],
                    trunc=n,
                    output_zero_level=False,
                    k=2,
                ),
            ),
        )

        g_voc = t_sum(
            core.tensor_fmexp(
                to_dense_first_on(g_w),
                dy_j[:P_g],
                trunc=m,
                output_zero_level=False,
                k=0,
            ),
            t_sum(
                core.tensor_fmexp(
                    to_dense_first_on(S_y),
                    dy_j[:P_g],
                    trunc=m,
                    output_zero_level=False,
                    k=1,
                ),
                core.tensor_fmexp(
                    to_dense_first_on(T_y),
                    dy_j[:P_g],
                    trunc=m,
                    output_zero_level=False,
                    k=2,
                ),
            ),
        )

        # ----- stage 1: use exact VOC as corrected corner increment
        df1 = t_sub(f_voc, f_sj)
        dg1 = t_sub(g_voc, g_w)

        f_ij = t_sum(
            f_sj,
            t_scal(t_sum(df0, df1), 0.5),
        )
        g_ij = t_sum(
            g_w,
            t_scal(t_sum(dg0, dg1), 0.5),
        )

        # ----- final scalar correction, same as -1
        F_c = forcing(u_p, f_ij, g_ij)
        u_ij = u_base + 0.25 * (F_sw + F_s + F_w + F_c)

        return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

    if _FREE_VOC_ORDER == -2:
        G_ij = t_inner(dx_i[:P], dy_j[:P])
        dx_adj_dy = adj_right(dx_i, dy_j, trunc=n)
        dy_adj_dx = adj_right(dy_j, dx_i, trunc=m)

        def to_dense_first_on(A):
            return (jnp.zeros_like(A[0][..., :1]),) + tuple(A)

        # Affine edge data:
        #   u00 = u_swj,  u01 = u_sj,  u10 = u_w
        #   f00 = f_swj,  f01 = f_sj
        #   g00 = g_swj,  g10 = g_w
        delta_x_u = u_w - u_swj
        delta_y_u = u_sj - u_swj

        delta_x_g = t_sub(g_w, g_swj)
        delta_y_f = t_sub(f_sj, f_swj)

        # First-on source terms
        S_x = t_sum(
            t_scal(dx_i[:P_f], u_sj),
            adj_left(g_swj, dx_i, trunc=n),
        )
        T_x = t_sum(
            t_scal(dx_i[:P_f], delta_x_u),
            adj_left(delta_x_g, dx_i, trunc=n),
        )

        S_y = t_sum(
            t_scal(dy_j[:P_g], u_w),
            adj_left(f_swj, dy_j, trunc=m),
        )
        T_y = t_sum(
            t_scal(dy_j[:P_g], delta_y_u),
            adj_left(delta_y_f, dy_j, trunc=m),
        )

        # Exact-exp VOC traces, returned again as first-on tuples
        f_ij = t_sum(
            core.tensor_fmexp(
                to_dense_first_on(f_sj),
                dx_i[:P_f],
                trunc=n,
                output_zero_level=False,
                k=0,  # exp
            ),
            t_sum(
                core.tensor_fmexp(
                    to_dense_first_on(S_x),
                    dx_i[:P_f],
                    trunc=n,
                    output_zero_level=False,
                    k=1,  # phi1
                ),
                core.tensor_fmexp(
                    to_dense_first_on(T_x),
                    dx_i[:P_f],
                    trunc=n,
                    output_zero_level=False,
                    k=2,  # phi2
                ),
            ),
        )

        g_ij = t_sum(
            core.tensor_fmexp(
                to_dense_first_on(g_w),
                dy_j[:P_g],
                trunc=m,
                output_zero_level=False,
                k=0,  # exp
            ),
            t_sum(
                core.tensor_fmexp(
                    to_dense_first_on(S_y),
                    dy_j[:P_g],
                    trunc=m,
                    output_zero_level=False,
                    k=1,  # phi1
                ),
                core.tensor_fmexp(
                    to_dense_first_on(T_y),
                    dy_j[:P_g],
                    trunc=m,
                    output_zero_level=False,
                    k=2,  # phi2
                ),
            ),
        )

        def forcing(u_val, f_val, g_val):
            return (
                    u_val * G_ij
                    + t_inner(f_val, dx_adj_dy)
                    + t_inner(g_val, dy_adj_dx)
            )

        F_sw = forcing(u_swj, f_swj, g_swj)
        F_s = forcing(u_sj, f_sj, g_sj)
        F_w = forcing(u_w, f_w, g_w)

        # NE corner with exact-exp VOC traces, only scalar u*G implicit
        B_ne = (
                t_inner(f_ij, dx_adj_dy)
                + t_inner(g_ij, dy_adj_dx)
        )

        u_base = u_sj + u_w - u_swj
        u_ij = (
                       u_base
                       + 0.25 * (F_sw + F_s + F_w + B_ne)
               ) / (1.0 - 0.25 * G_ij)

        return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

    if _FREE_VOC_ORDER == -1:
        G_ij = t_inner(dx_i[:P], dy_j[:P])
        dx_adj_dy = adj_right(dx_i, dy_j, trunc=n)
        dy_adj_dx = adj_right(dy_j, dx_i, trunc=m)

        def f_increment(u_val, f_val, g_val):
            return t_sum(
                t_sum(
                    t_scal(dx_i[:P_f], u_val),
                    t_prod(f_val, dx_i[:P_f], trunc=n),
                ),
                adj_left(g_val, dx_i, trunc=n),
            )

        def g_increment(u_val, f_val, g_val):
            return t_sum(
                t_sum(
                    t_scal(dy_j[:P_g], u_val),
                    t_prod(g_val, dy_j[:P_g], trunc=m),
                ),
                adj_left(f_val, dy_j, trunc=m),
            )

        def forcing(u_val, f_val, g_val):
            return (
                    u_val * G_ij
                    + t_inner(f_val, dx_adj_dy)
                    + t_inner(g_val, dy_adj_dx)
            )

        # stage 0: one-sided provisional edge advances
        df0 = f_increment(u_sj, f_sj, g_sj)
        dg0 = g_increment(u_w, f_w, g_w)

        f_p = t_sum(f_sj, df0)
        g_p = t_sum(g_w, dg0)

        F_sw = forcing(u_swj, f_swj, g_swj)
        F_s = forcing(u_sj, f_sj, g_sj)
        F_w = forcing(u_w, f_w, g_w)

        u_base = u_sj + u_w - u_swj
        u_p = u_base + F_sw

        # stage 1: coupled correction for f and g at the provisional corner
        df1 = f_increment(u_p, f_p, g_p)
        dg1 = g_increment(u_p, f_p, g_p)

        f_ij = t_sum(
            f_sj,
            t_scal(t_sum(df0, df1), 0.5),
        )
        g_ij = t_sum(
            g_w,
            t_scal(t_sum(dg0, dg1), 0.5),
        )

        # final scalar correction with corrected corner tensors
        F_c = forcing(u_p, f_ij, g_ij)
        u_ij = u_base + 0.25 * (F_sw + F_s + F_w + F_c)

        return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

    if _FREE_VOC_ORDER == 0:
        f_ij = t_sum(
            t_sum(
                f_sj,
                t_scal(dx_i[:P_f], u_sj),
            ),
            t_sum(
                t_prod(f_sj, dx_i[:P_f], trunc=n),
                adj_left(g_sj, dx_i, trunc=n),
            ),
        )

        g_ij = t_sum(
            t_sum(
                g_w,
                t_scal(dy_j[:P_g], u_w),
            ),
            t_sum(
                t_prod(g_w, dy_j[:P_g], trunc=m),
                adj_left(f_w, dy_j, trunc=m),
            ),
        )

        G_ij = t_inner(dx_i[:P], dy_j[:P])

        if M == 1 and N == 1:
            kap = 1.0 / 12.0
            G2_ij = G_ij * G_ij
            u_ij = (u_sj + u_w) * (1.0 + 0.5 * G_ij + kap * G2_ij) - u_swj * (1.0 - kap * G2_ij)
        else:
            dx_adj_dy = adj_right(dx_i, dy_j, trunc=n)
            dy_adj_dx = adj_right(dy_j, dx_i, trunc=m)

            def forcing(u_val, f_val, g_val):
                return (
                        u_val * G_ij
                        + t_inner(f_val, dx_adj_dy)
                        + t_inner(g_val, dy_adj_dx)
                )

            F_sw = forcing(u_swj, f_swj, g_swj)
            F_s = forcing(u_sj, f_sj, g_sj)
            F_w = forcing(u_w, f_w, g_w)

            u_p = u_sj + u_w - u_swj + F_sw
            F_p = forcing(u_p, f_ij, g_ij)
            u_ij = u_sj + u_w - u_swj + 0.25 * (F_sw + F_s + F_w + F_p)

        return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

    # _FREE_VOC_ORDER >= 1

    delta_x_u = u_w - u_swj
    delta_y_u = u_sj - u_swj
    bar_u = 0.5 * (u_w + u_sj)

    # reusable primitive expensive ops
    Px_sw = t_prod(f_swj, dx_i[:P_f], trunc=n)
    Px_s = t_prod(f_sj, dx_i[:P_f], trunc=n)

    Py_sw = t_prod(g_swj, dy_j[:P_g], trunc=m)
    Py_w = t_prod(g_w, dy_j[:P_g], trunc=m)

    Ax_sw = adj_left(g_swj, dx_i, trunc=n)
    Ax_w = adj_left(g_w, dx_i, trunc=n)

    Ay_sw = adj_left(f_swj, dy_j, trunc=m)
    Ay_s = adj_left(f_sj, dy_j, trunc=m)

    C_x = t_sum(
        t_scal(dx_i[:P_f], u_sj),
        t_sum(Px_s, Ax_sw),
    )
    C_y = t_sum(
        t_scal(dy_j[:P_g], u_w),
        t_sum(Py_w, Ay_sw),
    )

    D_x = t_sum(
        t_scal(dx_i[:P_f], delta_x_u),
        t_sub(Ax_w, Ax_sw),
    )
    D_y = t_sum(
        t_scal(dy_j[:P_g], delta_y_u),
        t_sub(Ay_s, Ay_sw),
    )

    M_x = t_sum(
        t_scal(dx_i[:P_f], bar_u),
        t_scal(
            t_sum(
                t_sum(Px_sw, Px_s),
                t_sum(Ax_sw, Ax_w),
            ),
            0.5,
        ),
    )
    M_y = t_sum(
        t_scal(dy_j[:P_g], bar_u),
        t_scal(
            t_sum(
                t_sum(Py_sw, Py_w),
                t_sum(Ay_sw, Ay_s),
            ),
            0.5,
        ),
    )

    F_1 = t_sum(C_x, t_scal(D_x, 0.5))
    G_1 = t_sum(C_y, t_scal(D_y, 0.5))

    F_2 = t_sum(
        t_prod(
            t_lin2(0.5, C_x, 1.0 / 6.0, D_x),
            dx_i[:P_f],
            trunc=n,
        ),
        adj_left(M_y, dx_i, trunc=n),
    )
    G_2 = t_sum(
        t_prod(
            t_lin2(0.5, C_y, 1.0 / 6.0, D_y),
            dy_j[:P_g],
            trunc=m,
        ),
        adj_left(M_x, dy_j, trunc=m),
    )

    f_ij = t_sum(f_sj, t_sum(F_1, F_2))
    g_ij = t_sum(g_w, t_sum(G_1, G_2))

    u_ij = u_sj + u_w - u_swj

    if _FREE_VOC_ORDER == 1:
        return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

    G_ij = t_inner(dx_i[:P], dy_j[:P])
    dx_adj_dy = adj_right(dx_i, dy_j, trunc=n)
    dy_adj_dx = adj_right(dy_j, dx_i, trunc=m)

    If_sw = t_inner(f_swj, dx_adj_dy)
    If_s = t_inner(f_sj, dx_adj_dy)
    Ig_sw = t_inner(g_swj, dy_adj_dx)
    Ig_w = t_inner(g_w, dy_adj_dx)

    Sigma_0 = (
            u_swj * G_ij
            + If_sw
            + Ig_sw
    )
    Sigma_x = (
            delta_x_u * G_ij
            + (Ig_w - Ig_sw)
    )
    Sigma_y = (
            delta_y_u * G_ij
            + (If_s - If_sw)
    )

    u_ij = u_ij + Sigma_0 + 0.5 * (Sigma_x + Sigma_y)

    if _FREE_VOC_ORDER == 2:
        return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

    u_ij = u_ij + (
            t_inner(
                t_lin2(0.5, M_x, -1.0 / 12.0, D_x),
                dx_adj_dy,
            )
            + t_inner(
        t_lin2(0.5, M_y, -1.0 / 12.0, D_y),
        dy_adj_dx,
    )
    )

    if _FREE_VOC_ORDER == 3:
        return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)

    N_x = t_sum(
        t_scal(M_x, 5.0 / 6.0),
        t_sum(
            t_scal(C_x, -1.0 / 3.0),
            t_scal(D_x, -1.0 / 3.0),
        ),
    )
    N_y = t_sum(
        t_scal(M_y, 5.0 / 6.0),
        t_sum(
            t_scal(C_y, -1.0 / 3.0),
            t_scal(D_y, -1.0 / 3.0),
        ),
    )

    K_x4 = t_sum(
        t_prod(
            t_lin2(1.0 / 6.0, M_x, -1.0 / 24.0, D_x),
            dx_i[:P_f],
            trunc=n,
        ),
        adj_left(N_y, dx_i, trunc=n),
    )
    K_y4 = t_sum(
        t_prod(
            t_lin2(1.0 / 6.0, M_y, -1.0 / 24.0, D_y),
            dy_j[:P_g],
            trunc=m,
        ),
        adj_left(N_x, dy_j, trunc=m),
    )

    u_ij = u_ij + (
            (G_ij / 12.0) * (3.0 * Sigma_0 + Sigma_x + Sigma_y)
            + t_inner(K_x4, dx_adj_dy)
            + t_inner(K_y4, dy_adj_dx)
    )

    return (u_ij, f_ij, g_ij), (u_ij, f_ij, g_ij)