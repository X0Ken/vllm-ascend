// Independent native prototype; does not replace either original operator.
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ops/empty.h>
#include "v41_op_api_common_workspace.h"
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
using Six = std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
Six allocate(const at::Tensor& residual) {
    auto n=residual.size(0);auto options=residual.options();
    return {at::empty({n,5120},options),at::empty({n,4},options.dtype(at::kFloat)),
            at::empty({n,4,4},options.dtype(at::kFloat)),at::empty({n,4},options.dtype(at::kFloat)),
            at::empty_like(residual),at::empty({n,5120},options.dtype(at::kFloat))};
}
Six post_pre_norm(const at::Tensor& residual,const at::Tensor& fn,const at::Tensor& scale,
    const at::Tensor& base,const at::Tensor& pre_mix,const at::Tensor& attn_x,
    const at::Tensor& attn_post,const at::Tensor& attn_comb,const at::Tensor& gamma,int64_t iters,double norm_eps,double hc_eps,double following_eps) {
    TORCH_CHECK(residual.dim()==3 && residual.size(0)>0 && residual.size(0)<=192 && residual.size(1)==4 && residual.size(2)==5120);
    auto n=residual.size(0);TORCH_CHECK(iters==20 && norm_eps>0 && hc_eps>0);
    TORCH_CHECK(residual.scalar_type()==at::kBFloat16 && residual.is_contiguous());
    TORCH_CHECK(attn_x.sizes()==at::IntArrayRef({n,5120}) && attn_x.scalar_type()==at::kBFloat16 && attn_x.is_contiguous());
    TORCH_CHECK(fn.sizes()==at::IntArrayRef({24,20480}) && scale.sizes()==at::IntArrayRef({3}) && base.sizes()==at::IntArrayRef({24}));
    TORCH_CHECK(pre_mix.sizes()==at::IntArrayRef({n,4}) && attn_post.sizes()==at::IntArrayRef({n,4}) && attn_comb.sizes()==at::IntArrayRef({n,4,4}));
    for (const auto& t : {fn,scale,base,pre_mix,attn_post,attn_comb}) {
        TORCH_CHECK(t.scalar_type()==at::kFloat && t.is_contiguous() && t.device()==residual.device());
    }
    TORCH_CHECK(attn_x.device()==residual.device());
    TORCH_CHECK(gamma.sizes()==at::IntArrayRef({5120}) && gamma.scalar_type()==at::kBFloat16 && gamma.is_contiguous() && gamma.device()==residual.device() && following_eps>0);
    auto result=allocate(residual);auto y=std::get<0>(result);auto post=std::get<1>(result);
    auto comb=std::get<2>(result);auto pre=std::get<3>(result);auto residual_out=std::get<4>(result);auto y32=std::get<5>(result);
    c10::optional<at::Tensor> mix=pre_mix;
    int64_t v41_hc_mult=4;
    EXEC_NPU_CMD(aclnnV41HcPostPreNorm,residual,fn,scale,base,mix,attn_x,attn_post,attn_comb,gamma,v41_hc_mult,iters,hc_eps,norm_eps,following_eps,y,post,comb,pre,residual_out,y32);
    return result;
}
Six allocate_symbolic(const at::Tensor& residual) {
    auto n=residual.sym_size(0);auto o=residual.options();
    return {at::empty_symint(std::vector<c10::SymInt>{n,5120},o),
            at::empty_symint(std::vector<c10::SymInt>{n,4},o.dtype(at::kFloat)),
            at::empty_symint(std::vector<c10::SymInt>{n,4,4},o.dtype(at::kFloat)),
            at::empty_symint(std::vector<c10::SymInt>{n,4},o.dtype(at::kFloat)),
            at::empty_like(residual),at::empty_symint(std::vector<c10::SymInt>{n,5120},o.dtype(at::kFloat))};
}
Six v41_binding_meta(const at::Tensor& residual,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,double,double,double) { return allocate_symbolic(residual); }
TORCH_LIBRARY(v41_postprenorm_native,m) {
    m.def("post_pre_norm(Tensor residual, Tensor fn, Tensor scale, Tensor base, Tensor pre_mix, Tensor attn_x, Tensor attn_post, Tensor attn_comb, Tensor gamma, int iters, float norm_eps, float hc_eps, float following_eps) -> (Tensor,Tensor,Tensor,Tensor,Tensor,Tensor)");
}
TORCH_LIBRARY_IMPL(v41_postprenorm_native,PrivateUse1,m) {m.impl("post_pre_norm",&post_pre_norm);}
TORCH_LIBRARY_IMPL(v41_postprenorm_native,Meta,m) {m.impl("post_pre_norm",&v41_binding_meta);}
