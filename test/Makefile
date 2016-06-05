# ----------------------------------------------------------------------------
# Makefile research application TestNeuralNet
# Version 2, 7 Sept 2015
# ---------------------------------------------------------------------------- 
# Note:
# $@ = File name of the target.
# $< = Name of the first dependency.
# .PHONY tells make that the target does not correspond to a real file.
# A '-' before a command tells make to ingnore errors.  eg. "-rm -f *.o"
# ----------------------------------------------------------------------------

PRODUCT=TestTfsDnn

# -----------------------------------------------------------------------------
# Directories:
# -----------------------------------------------------------------------------
BUILD_DIR	= ../../build/
INC_LIB_DIR	= ../../lib/
INC_DNN_DIR	= ../src/
LIB_DIR		= $(BUILD_DIR)lib/
BIN_DIR     = $(BUILD_DIR)bin/
OBJ_DIR     = $(BUILD_DIR)$(PRODUCT)_obj/
INSTALL_DIR = $(BIN_DIR)

# -----------------------------------------------------------------------------
# Compile and linker flags
# -----------------------------------------------------------------------------
# CC      = 
# CDEFS   =

CFLAGS  =  -Wall -O3
IFLAGS  = -I$(INC_LIB_DIR) -I$(INC_DNN_DIR)
LDFLAGS = 
COMPILE = g++ $(IFLAGS) $(CFLAGS) $(CDEFS) -c
LINK    = g++ -pthread

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------
TARGET = $(INSTALL_DIR)$(PRODUCT)

.PHONY: all
all:  $(TARGET)

# -----------------------------------------------------------------------------
# Libraries that we need
# -----------------------------------------------------------------------------
LIBS = $(LIB_DIR)tfs_dnn.a $(LIB_DIR)tfs_lib.a


# -----------------------------------------------------------------------------
# We list all of our .obj files here.
# -----------------------------------------------------------------------------
OBJS = 	$(OBJ_DIR)main.o\
        $(OBJ_DIR)TestMatrix.o\
		$(OBJ_DIR)test2D.o\
        $(OBJ_DIR)testBuilder.o\
        $(OBJ_DIR)testFullyConnected.o\
		$(OBJ_DIR)test2Layer.o\
        $(OBJ_DIR)testCifar10.o\
        $(OBJ_DIR)testIO.o

# -----------------------------------------------------------------------------
# We list the individual source file dependencies here.
# -----------------------------------------------------------------------------
$(OBJ_DIR)main.o                : main.cpp
$(OBJ_DIR)TestMatrix.o          : TestMatrix.cpp TestMatrix.hpp
$(OBJ_DIR)test2D.o              : test2D.cpp test2D.hpp
$(OBJ_DIR)testBuilder.o         : testBuilder.cpp testBuilder.hpp
$(OBJ_DIR)testFullyConnected.o  : testFullyConnected.cpp testFullyConnected.hpp
$(OBJ_DIR)test2Layer.o          : test2Layer.cpp test2Layer.hpp
$(OBJ_DIR)testCifar10.o         : testCifar10.cpp testCifar10.hpp
$(OBJ_DIR)testIO.o              : testIO.cpp testIO.hpp



# -----------------------------------------------------------------------------
# Make all of the objects dependent on this makefile.
# Recompile everything if we change this makefile.
# -----------------------------------------------------------------------------
$(OBJS): Makefile

# -----------------------------------------------------------------------------
# Compile pattern rule for making a .o from .cpp files for static
# -----------------------------------------------------------------------------
$(OBJ_DIR)%.o: %.cpp
	$(COMPILE) $< -o $@

# -----------------------------------------------------------------------------
# Link macro
# -----------------------------------------------------------------------------
$(TARGET): $(OBJS)
	@echo "Building target" $@ "..." 
	$(LINK) $(OBJS) $(LIBS) -o $@


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


