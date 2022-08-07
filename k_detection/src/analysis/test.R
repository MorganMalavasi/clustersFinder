sayHello <- function(){
   print('hello')
}

sayHello()

if(!require('fpc')) {
    install.packages('fpc')
    library('fpc')
}

set.seed(20000)
options(digits=3)
face <- rFace(100,dMoNo=2,dNoEy=0,p=8)
# print(face)