add_mlir_library(MLIRDistributeParallelismInterfaces
    DistributeParallelismInterfaces.cpp

    DEPENDS
    MLIRDistributeParallelismInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)