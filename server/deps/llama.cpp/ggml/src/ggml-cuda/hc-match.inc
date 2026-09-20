static int ggml_cuda_match_hc_mix(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_mix_args & args, enum ggml_op * ops) {
    if (i < 0 || i >= cgraph->n_nodes) { return 0; }
    const ggml_tensor * node = cgraph->nodes[i];
    if (!(node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID && node->type == GGML_TYPE_F32 &&
        i + 5 < cgraph->n_nodes &&
        cgraph->nodes[i + 1]->op == GGML_OP_MUL && cgraph->nodes[i + 2]->op == GGML_OP_RESHAPE &&
        cgraph->nodes[i + 3]->op == GGML_OP_VIEW && cgraph->nodes[i + 4]->op == GGML_OP_CONT)) {
        return 0;
    }
    const ggml_tensor * gate  = node->src[0];
    const ggml_tensor * mul   = cgraph->nodes[i + 1];
    const ggml_tensor * view0 = cgraph->nodes[i + 3];
    const ggml_tensor * cont  = cgraph->nodes[i + 4];
    const ggml_tensor * xn    = mul->src[0] == node ? mul->src[1] : mul->src[0];

    const int64_t n_embd   = cont->ne[0];
    const int64_t n_tokens = cont->ne[1];
    const int64_t hc_dim   = mul->ne[0];
    const int     hc       = n_embd > 0 ? (int) (hc_dim / n_embd) : 0;
    const size_t  row_size = hc_dim * sizeof(float);

    auto stream_view_ok = [&](const ggml_tensor * v, int c) {
        return v->op == GGML_OP_VIEW && v->view_src == mul && v->type == GGML_TYPE_F32 &&
            v->view_offs == (size_t) c * n_embd * sizeof(float) &&
            v->ne[0] == n_embd && v->ne[1] == n_tokens && v->ne[2] == 1 && v->ne[3] == 1 &&
            v->nb[0] == sizeof(float) && v->nb[1] == row_size;
    };

    bool ok = (mul->src[0] == node || mul->src[1] == node) && xn != node &&
        mul->type == GGML_TYPE_F32 && xn->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(xn) && ggml_is_contiguous(gate) && ggml_are_same_shape(xn, gate) &&
        ggml_are_same_shape(mul, xn) && ggml_are_same_shape(mul, node) &&
        mul->ne[2] == 1 && mul->ne[3] == 1 && ggml_nrows(mul) == n_tokens &&
        hc >= 2 && hc <= 16 && hc_dim == (int64_t) hc * n_embd &&
        cgraph->nodes[i + 2]->view_src == mul && cont->src[0] == view0 && stream_view_ok(view0, 0) &&
        cont->type == GGML_TYPE_F32 && ggml_is_contiguous(cont) && cont->ne[2] == 1 && cont->ne[3] == 1;

    const int n_ops = ok ? 5 + 2 * (hc - 1) + 1 : 0;
    ok = ok && i + n_ops <= cgraph->n_nodes;
    if (!ok) {
        return 0;
    }

    ops[0] = GGML_OP_UNARY; ops[1] = GGML_OP_MUL; ops[2] = GGML_OP_RESHAPE; ops[3] = GGML_OP_VIEW; ops[4] = GGML_OP_CONT;
    const ggml_tensor * prev = cont;
    for (int c = 1; ok && c < hc; ++c) {
        const ggml_tensor * v   = cgraph->nodes[i + 5 + 2 * (c - 1)];
        const ggml_tensor * add = cgraph->nodes[i + 6 + 2 * (c - 1)];
        ok = stream_view_ok(v, c) && add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 &&
            add->src[0] == prev && add->src[1] == v && ggml_are_same_shape(add, cont);
        ops[5 + 2 * (c - 1)] = GGML_OP_VIEW;
        ops[6 + 2 * (c - 1)] = GGML_OP_ADD;
        prev = add;
    }
    ggml_tensor * scale = cgraph->nodes[i + n_ops - 1];
    ops[n_ops - 1] = GGML_OP_SCALE;
    ok = ok && scale->op == GGML_OP_SCALE && scale->src[0] == prev && scale->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(scale) && ggml_are_same_shape(scale, cont);
    if (!ok) {
        return 0;
    }

    args.xn    = xn;
    args.gate  = gate;
    args.dst   = scale;
    args.hc    = hc;
    args.scale = ggml_get_op_params_f32(scale, 0);
    args.bias  = ggml_get_op_params_f32(scale, 1);
    return n_ops;
}

static int ggml_cuda_hc_mix_closed(const ggml_cgraph * graph, int index, ggml_cuda_hc_mix_args & args) {
    ggml_op ops[36];
    const int count=ggml_cuda_match_hc_mix(graph,index,args,ops);
    if (count==0) { return 0; }
    int indices[36];
    for (int j=0;j<count;++j) { indices[j]=index+j; }
    const int output=index+count-1;
    return ggml_can_fuse_subgraph_ext(graph,indices,count,ops,&output,1) ? count : 0;
}
