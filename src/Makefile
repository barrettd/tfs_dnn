# ----------------------------------------------------------------------------
# Makefile research static library: tfs_dnn.a
# Version 1, 7 May 2016
# ---------------------------------------------------------------------------- 
# Note:
# $@ = File name of the target.
# $< = Name of the first dependency.
# .PHONY tells make that the target does not correspond to a real file.
# A '-' before a command tells make to ingnore errors.  eg. "-rm -f *.o"
# ----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Directories:
# -----------------------------------------------------------------------------
BUILD_DIR   = ../../build/
INC_LIB_DIR = ../../lib/
LIB_DIR	    = $(BUILD_DIR)lib/
BIN_DIR     = $(BUILD_DIR)bin/
OBJ_DIR     = $(BUILD_DIR)DNN_obj/
INSTALL_DIR = $(LIB_DIR)

# -----------------------------------------------------------------------------
# Compile and linker flags
# -----------------------------------------------------------------------------
# CC      = 
# CDEFS   =

CFLAGS  = -Wall -O3
IFLAGS  = -I $(INC_LIB_DIR)
LDFLAGS = 
COMPILE_DYNAMIC  = $(CC) $(IFLAGS) $(CFLAGS) -fpic $(CDEFS) -c
COMPILE_STATIC   = g++ $(IFLAGS) $(CFLAGS) $(CDEFS) -c

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
TARGET = $(INSTALL_DIR)tfs_dnn.a

.PHONY: all
all:  $(TARGET)

# -----------------------------------------------------------------------------
# Libraries that we need
# -----------------------------------------------------------------------------
# LIBS = $(LIB_DIR)tfs_lib.a
LIBS =

# -----------------------------------------------------------------------------
# We list all of our .obj files here.
# -----------------------------------------------------------------------------
OBJS = 	$(OBJ_DIR)Dnn.o\
        $(OBJ_DIR)DnnBuilder.o\
        $(OBJ_DIR)DnnLayer.o\
        $(OBJ_DIR)DnnLayerConvolution.o\
        $(OBJ_DIR)DnnLayerDropout.o\
        $(OBJ_DIR)DnnLayerFullyConnected.o\
        $(OBJ_DIR)DnnLayerInput.o\
        $(OBJ_DIR)DnnLayerLocalResponseNormalization.o\
        $(OBJ_DIR)DnnLayerMaxout.o\
        $(OBJ_DIR)DnnLayerPool.o\
        $(OBJ_DIR)DnnLayerRectifiedLinearUnit.o\
        $(OBJ_DIR)DnnLayerRegression.o\
        $(OBJ_DIR)DnnLayerSigmoid.o\
        $(OBJ_DIR)DnnLayerSoftmax.o\
        $(OBJ_DIR)DnnLayerSupportVectorMachine.o\
        $(OBJ_DIR)DnnLayerTanh.o\
        $(OBJ_DIR)DnnTrainer.o\
        $(OBJ_DIR)DnnTrainerAdaDelta.o\
        $(OBJ_DIR)DnnTrainerAdam.o\
        $(OBJ_DIR)DnnTrainerSGD.o\
        $(OBJ_DIR)Error.o\
        $(OBJ_DIR)Utility.o\
        $(OBJ_DIR)CsvFile.o



# -----------------------------------------------------------------------------
# We list the individual source file dependencies here.
# -----------------------------------------------------------------------------
$(OBJ_DIR)Dnn.o                                 : Dnn.cpp Dnn.h
$(OBJ_DIR)DnnBuilder.o                          : DnnBuilder.cpp DnnBuilder.h
$(OBJ_DIR)DnnLayer.o                            : DnnLayer.cpp DnnLayer.h
$(OBJ_DIR)DnnLayerConvolution.o                 : DnnLayerConvolution.cpp DnnLayerConvolution.h
$(OBJ_DIR)DnnLayerDropout.o                     : DnnLayerDropout.cpp DnnLayerDropout.h
$(OBJ_DIR)DnnLayerFullyConnected.o              : DnnLayerFullyConnected.cpp DnnLayerFullyConnected.h
$(OBJ_DIR)DnnLayerInput.o                       : DnnLayerInput.cpp DnnLayerInput.h
$(OBJ_DIR)DnnLayerLocalResponseNormalization.o  : DnnLayerLocalResponseNormalization.cpp DnnLayerLocalResponseNormalization.h
$(OBJ_DIR)DnnLayerMaxout.o                      : DnnLayerMaxout.cpp DnnLayerMaxout.h
$(OBJ_DIR)DnnLayerPool.o                        : DnnLayerPool.cpp DnnLayerPool.h
$(OBJ_DIR)DnnLayerRectifiedLinearUnit.o         : DnnLayerRectifiedLinearUnit.cpp DnnLayerRectifiedLinearUnit.h
$(OBJ_DIR)DnnLayerRegression.o                  : DnnLayerRegression.cpp DnnLayerRegression.h
$(OBJ_DIR)DnnLayerSigmoid.o                     : DnnLayerSigmoid.cpp DnnLayerSigmoid.h
$(OBJ_DIR)DnnLayerSoftmax.o                     : DnnLayerSoftmax.cpp DnnLayerSoftmax.h
$(OBJ_DIR)DnnLayerSupportVectorMachine.o        : DnnLayerSupportVectorMachine.cpp DnnLayerSupportVectorMachine.h
$(OBJ_DIR)DnnLayerTanh.o                        : DnnLayerTanh.cpp DnnLayerTanh.h
$(OBJ_DIR)DnnTrainer.o                          : DnnTrainer.cpp DnnTrainer.h
$(OBJ_DIR)DnnTrainerAdaDelta.o                  : DnnTrainerAdaDelta.cpp DnnTrainerAdaDelta.h
$(OBJ_DIR)DnnTrainerAdam.o                      : DnnTrainerAdam.cpp DnnTrainerAdam.h
$(OBJ_DIR)DnnTrainerSGD.o                       : DnnTrainerSGD.cpp DnnTrainerSGD.h
$(OBJ_DIR)Error.o                               : Error.cpp Error.h
$(OBJ_DIR)Utility.o                             : Utility.cpp Utility.h
$(OBJ_DIR)CsvFile.o                             : CsvFile.cpp CsvFile.h


# -----------------------------------------------------------------------------
# Make all of the objects dependent on this makefile.
# Recompile everything if we change this makefile.
# -----------------------------------------------------------------------------
$(OBJS): Makefile

# -----------------------------------------------------------------------------
# Compile pattern rule for making a .o from .cpp files for static
# -----------------------------------------------------------------------------
$(OBJ_DIR)%.o: %.cpp
	$(COMPILE_STATIC) $< -o $@ 

# -----------------------------------------------------------------------------
# Link macro for static
# -----------------------------------------------------------------------------
$(TARGET): $(OBJS) 
	ar rvsc $@ $(OBJS) $(LIBS)

# -----------------------------------------------------------------------------
# Targets for making the build directories
# -----------------------------------------------------------------------------
.PHONY: directories
directories:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(LIB_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(OBJ_DIR)
	mkdir -p $(INSTALL_DIR)

	
# -----------------------------------------------------------------------------
# Targets for erasing intermediate and release files.
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	-rm -f *.out *.[oa] $(OBJS) *.[oa] \
           *.bin *~ *.bak core *.utf8 .#*

.PHONY: cleanall
cleanall: clean
	-rm -f $(TARGET) 


