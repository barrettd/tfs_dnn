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
BUILD_DIR   = ../build/
INC_LIB_DIR = ../lib/
LIB_DIR	    = $(BUILD_DIR)lib/
BIN_DIR     = $(BUILD_DIR)bin/
OBJ_DIR     = $(BUILD_DIR)DNN_obj/
INSTALL_DIR = $(LIB_DIR)

# -----------------------------------------------------------------------------
# Compile and linker flags
# -----------------------------------------------------------------------------
# CC      = 
# CDEFS   =

CFLAGS  = 
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
OBJS = 	$(OBJ_DIR)NeuralNet.o\
		$(OBJ_DIR)NeuralNetLayer.o\
		$(OBJ_DIR)Neuron.o\
		$(OBJ_DIR)NeuronConnection.o\
		$(OBJ_DIR)NNAutoPilot.o\
		$(OBJ_DIR)DataSource.o\
		$(OBJ_DIR)FannData.o\
		$(OBJ_DIR)NNData.o\
		$(OBJ_DIR)NNUtils.o\
		$(OBJ_DIR)nnfstream.o

# -----------------------------------------------------------------------------
# We list the individual source file dependencies here.
# -----------------------------------------------------------------------------
$(OBJ_DIR)NeuralNet.o        : NeuralNet.cpp        NeuralNet.h 
$(OBJ_DIR)NeuralNetLayer.o   : NeuralNetLayer.cpp   NeuralNetLayer.h NeuralNet.h
$(OBJ_DIR)Neuron.o           : Neuron.cpp           Neuron.h NeuronConnection.h $(INC_LIB_DIR)random.h
$(OBJ_DIR)NeuronConnection.o : NeuronConnection.cpp NeuronConnection.h Neuron.h
$(OBJ_DIR)NNAutoPilot.o      : NNAutoPilot.cpp      NNAutoPilot.h
$(OBJ_DIR)DataSource.o       : DataSource.cpp       DataSource.h
$(OBJ_DIR)FannData.o         : FannData.cpp         FannData.h DataSource.h
$(OBJ_DIR)NNData.o           : NNData.cpp           NNData.h DataSource.h
$(OBJ_DIR)NNUtils.o          : NNUtils.cpp          NNUtils.h NNData.h DataSource.h
$(OBJ_DIR)nnfstream.o		 : nnfstream.cpp        nnfstream.h


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


