# pytorch-autojit

A stupidly-simple way to use pytorch's jit compiler.

This should be considered **only a proof-of-concept** right now.  It seems to work in some situations, but is brittle and will cause incorrect results if you're not careful.  You must heed and take seriously any warning messages that are printed during JIT-tracing.  Also it's very opinionated about the kinds of parameters that can be passed into a function.  

But, with all that said, here's all you have to do to try it:

```py
from autojit import autojit

@autojit
def fancy_box_score(stride, anchor, score, loc, hindex, windex, variances):
    axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
    priors = torch.cat([axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]).unsqueeze(0)
    box = decode(loc,priors,variances)
    x1,y1,x2,y2 = box[0]*1.0
    return (x1,y1,x2,y2,score)
```

Try using it with [timebudget](https://pypi.org/project/timebudget/) to see if/how much it's helping.  e.g.

```py
@timebudget
@autojit
def fancy_box_score(stride, anchor, score, loc, hindex, windex, variances):
    ...
```

