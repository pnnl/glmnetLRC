### glmnetLRC

#### Lasso and Elastic-Net Logistic Regression Classification with an Arbitrary Loss Function

Sego LH, Venzin AM, Ramey JA. 2016. glmnetLRC: Lasso and Elastic-Net Logistic Regression Classification (LRC) 
with an Arbitrary Loss Function in R. Pacific Northwest National Laboratory. http://github.com/pnnl/glmnetLRC.

The [package vignette](http://NEEDLINK) includes examples to help you get started, as well as mathematical 
details of the algorithms used by the package.

#### Installation instructions

Begin by installing dependencies from [CRAN](http://cran.r-project.org):

    install.packages(c("devtools", "glmnet", "plyr"))

The `Smisc` package (a dependency of `glmnetLRC`) contains C code and require compilation. To do this
* on a Mac, you'll need [Xcode](https://developer.apple.com/xcode/) 
* on Windows, you'll need to install [R tools](http://cran.r-project.org/bin/windows/Rtools/)
* on Linux, compilation should take place "automatically"

With the compilation tools in place, you can now install the `Smisc` and `glmnetLRC` packages
from [the PNNL github site](http://github.com/pnnl):

    devtools::install_github("pnnl/Smisc")
    devtools::install_github("pnnl/glmnetLRC")

#### Getting started

The vignette for the `glmnetLRC` package is the principal resource for understanding what the package does.  After installing
the package, you can can browse the package and the vignette as follows:

    library(glmnetLRC)
    browseVignettes("glmnetLRC")

And a list of all the package functions can be found this way:

    help(package = glmnetLRC)
    
And this will provide citation information:

    citation("glmnetLRC")

### Contributing

We welcome contributions to this package.  Please follow [these steps](http://pnnl.github.io/prepPackage) when contributing.
