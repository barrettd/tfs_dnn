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

PRODUCT=TestNeuralNet

# -----------------------------------------------------------------------------
# Directories:
# -----------------------------------------------------------------------------
BUILD_DIR	= ../build/
INC_LIB_DIR	= ../lib/
INC_NN_DIR	= ../NeuralNet/
LIB_DIR		= $(BUILD_DIR)lib/
BIN_DIR     = $(BUILD_DIR)bin/
OBJ_DIR     = $(BUILD_DIR)$(PRODUCT)_obj/
INSTALL_DIR = $(BIN_DIR)

# -----------------------------------------------------------------------------
# Compile and linker flags
# -----------------------------------------------------------------------------
# CC      = 
# CDEFS   =

CFLAGS  =  -Wall
IFLAGS  = -I$(INC_LIB_DIR) -I$(INC_NN_DIR)
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
LIBS = $(LIB_DIR)tfs_nn.a $(LIB_DIR)tfs_lib.a

# -----------------------------------------------------------------------------
# We list all of our .obj files here.
# -----------------------------------------------------------------------------
OBJS = 	$(OBJ_DIR)main.o\
		$(OBJ_DIR)SandBox.o\
		$(OBJ_DIR)TestAbelone.o\
		$(OBJ_DIR)TestContext.o\
		$(OBJ_DIR)TestMushroom.o\
		$(OBJ_DIR)TestRobot.o\
		$(OBJ_DIR)TwoSpiral.o\
		$(OBJ_DIR)TestXOR.o

# -----------------------------------------------------------------------------
# We list the individual source file dependencies here.
# -----------------------------------------------------------------------------
$(OBJ_DIR)main.o       		: main.cpp TestAbelone.h TestContext.h TestMushroom.h TestRobot.h TestXOR.h TwoSpiral.h
$(OBJ_DIR)SandBox.o		    : SandBox.cpp       SandBox.h
$(OBJ_DIR)TestAbelone.o		: TestAbelone.cpp   TestAbelone.h SandBox.h
$(OBJ_DIR)TestContext.o		: TestContext.cpp	TestContext.h SandBox.h
$(OBJ_DIR)TestMushroom.o	: TestMushroom.cpp	TestMushroom.h SandBox.h
$(OBJ_DIR)TestRobot.o		: TestRobot.cpp		TestRobot.h SandBox.h
$(OBJ_DIR)TwoSpiral.o		: TwoSpiral.cpp		TwoSpiral.h SandBox.h
$(OBJ_DIR)TestXOR.o			: TestXOR.cpp		TestXOR.h SandBox.h


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


