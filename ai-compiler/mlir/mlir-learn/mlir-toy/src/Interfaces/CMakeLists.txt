add_mlir_library(MLIRDistributeParallelismOpInterfaces
    DistributeParallelismOpInterfaces.cpp

    DEPENDS
    MLIRDistributeParallelismOpInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)