"""
Paradigm 1.0.226 - OpenCL enqueue / dispatch surface (extracted from add-on archive)

Sources (no blender addon used; nothing else from compute/ except fluidSim.cl):
  * houdini/scripts/python/PARADIGM_1_0/core.py        (add-on Python)
  * houdini21.0.700_windows_x86_64_cl19.42/dso/*.dll   (compiled DSO host code)
  * compute/openCL/fluidSim.cl                         (OpenCL kernel source)

DSO imports + calls the full OpenCL dispatch API (clCreateKernel, clSetKernelArg,
clEnqueueNDRangeKernel, clBuildProgram). Python add-on layer exposes a
user-facing custom-kernel injection surface via two CoreSolverNode methods.
"""

# =====================================================================
# Section 1 - OpenCL dispatch API surface (DSO import table)
# =====================================================================

DSO_OPENCL_DISPATCH_APIS = [
    "clBuildProgram",
    "clCreateKernel",
    "clEnqueueNDRangeKernel",
    "clEnqueueReadBuffer",
    "clEnqueueWriteBuffer",
    "clFinish",
    "clFlush",
    "clSetKernelArg",
]

# Dispatch flow:
#   clBuildProgram         - compile fluidSim.cl for device
#   clCreateKernel         - obtain cl_kernel by name
#   clSetKernelArg         - bind __global / __constant args
#   clEnqueueNDRangeKernel - the enqueue
#   clEnqueueReadBuffer / clEnqueueWriteBuffer - device <-> host sync
#   clFinish / clFlush     - drain command queue

# =====================================================================
# Section 2 - DSO diagnostic / log strings (runtime dispatch trace)
# =====================================================================

DSO_DISPATCH_DIAGNOSTIC_STRINGS = [
    "- Dispatch Wait For Command Queue:",
    "- Dispatch Work",
    "- Wait For Command Queue:",
    "COMPUTE::Manager::createCommandQueue() -> A command queue is already created, to create a new command queue you must reset!",
    "Create Command Queue",
    "Enqueue Compute Kernel -",
    "Enqueue Copy To Buffer -",
    "Enqueue Copy To Host -",
    "Failed to create command queue!",
    "Failed to set device when creating the command queue.",
    "Invalid command queue.",
    "Invalid kernel name.",
    "Kernel Buffer Sync",
    "Kernel Name:",
    "Set Kernel Buffer Argument",
    "Set Kernel Buffer Argument -",
    "clEnqueueMapBuffer",
    "clEnqueueNDRangeKernel",
    "clEnqueueReadBuffer",
    "clEnqueueUnmapMemObject",
    "clEnqueueWriteBuffer",
    "could not be used because it threw an error when initializing the compute command queues.",
    "total threads dispatched.",
]

# Dispatch sequence: Create Command Queue -> Wait For Command Queue ->
# Dispatch Work -> Dispatch Wait For Command Queue -> Enqueue Copy To Buffer/Host ->
# Enqueue Compute Kernel. Validation: Invalid kernel name / Invalid command queue.

# =====================================================================
# Section 3 - Python-side dispatch surface (verbatim from core.py)
# =====================================================================

class CoreSolverNode:
    """
    Houdini add-on class (PARADIGM_1_0/core.py). Host-side enqueue surface.
    """

    c_node = None

    # verbatim copy of computeCustomOpenClParametersToDictonary from core.py lines 607-760
    def computeCustomOpenClParametersToDictonary(self) -> str:

        if not self.checkHoudiniCompatibility():
        
            return ""
        
        l_data = {}

        l_data["result"] = []
        l_data["success"] = True 
        l_data["error"] = "" 

        l_CustomOpenClsParameter = self.c_node.parm('computeCustomOpenClSnippets')

        if l_CustomOpenClsParameter is None:

            l_data["result"] = []
            l_data["success"] = False 
            l_data["error"] = "Missing the {} parameter!".format("computeCustomOpenClSnippets")

            return utilities.standardOutputToString(l_data)

        l_keys = ["enable", "type", "order", "runOver", "kernelName", "code"]

        l_argumentKeys = ["mode", "field", "fieldPermissions", "constantType", "constantName", "constantValueFloat", "constantValueInt"]

        for i in range(l_CustomOpenClsParameter.evalAsInt()):

            l_indexSuffix = str(i + 1)

            l_currentData = {}

            for l_key in l_keys:

                l_parameterName = "computeCustomOpenCl{}{}".format(utilities.capitalizeFirstCharacter(l_key), l_indexSuffix)

                l_argumentParameter = self.c_node.parm(l_parameterName)

                if l_argumentParameter is None:

                    l_data["result"] = []
                    l_data["success"] = False 
                    l_data["error"] = "The {} parameter is missing!".format(l_parameterName)

                    return utilities.standardOutputToString(l_data)
                
                l_parameterType = l_argumentParameter.parmTemplate().type()

                if l_parameterType is None:

                    l_data["result"] = []
                    l_data["success"] = False 
                    l_data["error"] = "The {} parameter type is missing!".format(l_parameterName)

                    return utilities.standardOutputToString(l_data)
                
                if l_parameterType == hou.parmTemplateType.Toggle:

                    l_currentData[l_key] = bool(l_argumentParameter.evalAsInt())

                elif l_parameterType == hou.parmTemplateType.Int or l_parameterType == hou.parmTemplateType.Menu:

                    l_currentData[l_key] = l_argumentParameter.evalAsInt()

                elif l_parameterType == hou.parmTemplateType.Float:

                    l_currentData[l_key] = l_argumentParameter.evalAsFloat()

                elif l_parameterType == hou.parmTemplateType.String:

                    l_currentData[l_key] = l_argumentParameter.evalAsString()

                else:

                    l_data["result"] = []
                    l_data["success"] = False 
                    l_data["error"] = "The {} parameter type is unknown! ({})".format(l_argumentParameterName, str(l_parameterType))

                    return utilities.standardOutputToString(l_data)

            l_parameterName = "computeCustomOpenClArguments{}".format(l_indexSuffix)  

            l_customOpenClsArgumentsParameter = self.c_node.parm(l_parameterName)

            if l_customOpenClsArgumentsParameter is None:

                l_data["result"] = []
                l_data["success"] = False 
                l_data["error"] = "Missing the {} parameter!".format(l_parameterName)

                return utilities.standardOutputToString(l_data)

            l_currentData["arguments"] = []

            for a in range(l_customOpenClsArgumentsParameter.evalAsInt()):

                l_argumentsIndexSuffix = str(a + 1)

                l_currentData["arguments"].append({})

                for l_argumentKey in l_argumentKeys:

                    l_argumentParameterName = "computeCustomOpenClArguments{}{}{}".format(l_indexSuffix, utilities.capitalizeFirstCharacter(l_argumentKey), l_argumentsIndexSuffix)

                    l_argumentParameter = self.c_node.parm(l_argumentParameterName)

                    if l_argumentParameter is None:

                        l_data["result"] = []
                        l_data["success"] = False 
                        l_data["error"] = "The {} parameter is missing!".format(l_argumentParameterName)

                        return utilities.standardOutputToString(l_data)
                    
                    l_argumentParameterType = l_argumentParameter.parmTemplate().type()

                    if l_argumentParameterType is None:

                        l_data["result"] = []
                        l_data["success"] = False 
                        l_data["error"] = "The {} parameter type is missing!".format(l_argumentParameterName)

                        return utilities.standardOutputToString(l_data)
                    
                    if l_argumentParameterType == hou.parmTemplateType.Toggle:

                        l_currentData["arguments"][a][l_argumentKey] = bool(l_argumentParameter.evalAsInt())

                    elif l_argumentParameterType == hou.parmTemplateType.Int or l_argumentParameterType == hou.parmTemplateType.Menu:

                        l_currentData["arguments"][a][l_argumentKey] = l_argumentParameter.evalAsInt()

                    elif l_argumentParameterType == hou.parmTemplateType.Float:

                        l_currentData["arguments"][a][l_argumentKey] = l_argumentParameter.evalAsFloat()

                    elif l_argumentParameterType == hou.parmTemplateType.String:

                        l_currentData["arguments"][a][l_argumentKey] = l_argumentParameter.evalAsString()

                    else:

                        l_data["result"] = []
                        l_data["success"] = False 
                        l_data["error"] = "The {} parameter type is unknown! ({})".format(l_argumentParameterName, str(l_argumentParameterType))

                        return utilities.standardOutputToString(l_data)

            l_data["result"].append(l_currentData)

            return utilities.standardOutputToString(l_data)

        return utilities.standardOutputToString(l_data)
    

    # verbatim copy of computeCustomOpenClCreateKernel from core.py lines 761-1000
    def computeCustomOpenClCreateKernel(self, a_calledParameter, a_onlyUpdateHeader: bool = False):

        l_data = self.computeCustomOpenClParametersToDictonary()

        l_data = utilities.stringToStandardOutput(l_data)

        if not l_data["success"]:

            houdini.popUpMessage(l_data["error"])

            return

        l_calledParameterName = a_calledParameter.name()

        l_calledParameterDigits = utilities.getTrailingDigits(l_calledParameterName)

        if l_calledParameterDigits == "":

            houdini.popUpMessage("Unknow button triggered kernel creation!")

            return

        l_index = int(utilities.getTrailingDigits(l_calledParameterDigits))

        #Minus 1 because the Houdini multi-parameters start at 1 but lists in Python start at 0.
        l_entry = l_data["result"][l_index - 1]

        l_codeParameterName = "computeCustomOpenCl{}{}".format(utilities.capitalizeFirstCharacter("code"), l_index)

        l_codeParameter = self.c_node.parm(l_codeParameterName)

        if l_codeParameter is None:

            houdini.popUpMessage("Missing the {} parameter!".format(l_codeParameterName))

            return 
        
        l_output = "__kernel void {}(".format(l_entry["kernelName"])
        l_output += "\n"
        l_output += "\n"

        l_enums = __enum__.getValues()

        l_modeEnum = utilities.flipDictionary(l_enums["CORE::Solver::CustomOpenClArgumentMode"])
        l_fieldPermissionsEnum = utilities.flipDictionary(l_enums["CORE::Solver::CustomOpenClArgumentFieldPermissions"])
        l_constantTypeEnum = utilities.flipDictionary(l_enums["CORE::Solver::CustomOpenClArgumentConstantType"])
        l_runOverEnum = utilities.flipDictionary(l_enums["CORE::Solver::CustomOpenClRunOver"])

        l_voxelTypeWriteIncluded = False
        l_voxelTypeReadIncluded = False

        for l_argument in l_entry["arguments"]:

            l_mode = l_argument["mode"]

            if l_modeEnum[str(l_mode)].lower() == "field":    

                l_field = l_argument["field"]
                l_fieldPermissions = l_argument["fieldPermissions"]

                l_argumentType = "field_t"

                l_isVoxelType = False

                if l_field == "voxelType":

                    l_argumentType = "voxelType_t"

                    l_isVoxelType = True

                elif l_field == "particlePosition":

                    l_argumentType = "float"

                l_suffixes = [""]

                if "velocity" in l_field.lower() or "position" in l_field.lower():

                    l_suffixes = ["_x", "_y", "_z"]

                elif "color" in l_field.lower():

                    l_suffixes = ["_r", "_g", "_b"]

                if l_fieldPermissionsEnum[str(l_fieldPermissions)] == "Write" or l_fieldPermissionsEnum[str(l_fieldPermissions)] == "ReadWrite":

                    if l_isVoxelType == True:

                        l_voxelTypeWriteIncluded = True

                    for l_suffix in l_suffixes:

                        l_output += "__global {} * a_{}Write{},".format(l_argumentType, l_field, l_suffix)
                        l_output += "\n"

                if l_fieldPermissionsEnum[str(l_fieldPermissions)] == "Read" or l_fieldPermissionsEnum[str(l_fieldPermissions)] == "ReadWrite":

                    if l_isVoxelType == True:

                        l_voxelTypeReadIncluded = True

                    for l_suffix in l_suffixes:

                        l_output += "__global {} * a_{}Read{},".format(l_argumentType, l_field, l_suffix)
                        l_output += "\n"

            elif l_modeEnum[str(l_mode)].lower() == "constant":

                l_constantType = l_constantTypeEnum[str(l_argument["constantType"])].lower()
                l_constantName = l_argument["constantName"]

                l_output += "{} a_{},".format(l_constantType, l_constantName)
                l_output += "\n"

        l_output += "__constant struct CoreParametersOpenClStandardValues * a_standardValues,"
        l_output += "\n"
        l_output += "__constant struct CoreParametersSparseOctree * a_sparseOctree,"
        l_output += "\n"
        l_output += "__constant int * a_sparseOctreeRootSubElements,"
        l_output += "\n"
        l_output += "__global int * a_sparseOctreeSubElements,"
        l_output += "\n"
        l_output += "__global int * a_sparseOctreeStart"
        l_output += "\n"
        
        l_output += "\n"
        l_output += ") {"

        if a_onlyUpdateHeader:

            l_exitingCodeSplit = utilities.splitByCodeHeader(l_codeParameter.evalAsString())

            if len(l_exitingCodeSplit) == 2:

                l_exitingCode = l_exitingCodeSplit[1]

                l_output += l_exitingCode

                l_codeParameter.set(l_output) 

                return

        l_output += "\n"
        l_output += "\n"

        l_tabString = "    "

        l_voxelTypeSkipCollision = ""

        if l_voxelTypeReadIncluded:

            l_voxelTypeSkipCollision = "if(a_voxelTypeRead[l_index] & Bit_Core_VoxelType_Collision) return;"

        elif l_voxelTypeWriteIncluded:

            l_voxelTypeSkipCollision = "if(a_voxelTypeWrite[l_index] & Bit_Core_VoxelType_Collision) return;"

        if l_runOverEnum[str(l_entry["runOver"])] == "VoxelTile":

            l_output += l_tabString + "int l_x = get_local_id(0);"
            l_output += "\n"
            l_output += l_tabString + "int l_y = get_global_id(1);"
            l_output += "\n"
            l_output += l_tabString + "int l_z = get_global_id(2);"
            l_output += "\n"
            l_output += "\n"
            l_output += l_tabString + "int l_leafIndex = get_group_id(0);"
            l_output += "\n"
            l_output += "\n"
            l_output += l_tabString + "index_t l_index = coreGetVoxelIndexLocal(l_leafIndex, l_x, l_y, l_z);"
            l_output += "\n"

            if l_voxelTypeSkipCollision != "":

                l_output += "\n"
                l_output += l_tabString + l_voxelTypeSkipCollision
                l_output += "\n"

            l_output += "\n"
            l_output += l_tabString + "int l_start_x, l_start_y, l_start_z;"
            l_output += "\n"
            l_output += l_tabString + "coreGetStartVectorL0(l_leafIndex, &l_start_x, &l_start_y, &l_start_z, a_sparseOctreeStart);"
            l_output += "\n"
            l_output += "\n"
            l_output += l_tabString + "l_x += l_start_x;"
            l_output += "\n"
            l_output += l_tabString + "l_y += l_start_y;"
            l_output += "\n"
            l_output += l_tabString + "l_z += l_start_z;"
            l_output += "\n"
            l_output += "\n"
            l_output += l_tabString + "float l_worldPosition_x = (float)(l_x) * Const_Core_OctreeVoxelDivisionSize;"
            l_output += "\n"
            l_output += l_tabString + "float l_worldPosition_y = (float)(l_y) * Const_Core_OctreeVoxelDivisionSize;"
            l_output += "\n"
            l_output += l_tabString + "float l_worldPosition_z = (float)(l_z) * Const_Core_OctreeVoxelDivisionSize;"
            l_output += "\n"

        elif l_runOverEnum[str(l_entry["runOver"])] == "VoxelLinear":

            l_output += l_tabString + "index_t l_index = get_global_id(0);"
            l_output += "\n"

            if l_voxelTypeSkipCollision != "":

                l_output += "\n"
                l_output += l_tabString + l_voxelTypeSkipCollision
                l_output += "\n"

        elif l_runOverEnum[str(l_entry["runOver"])] == "Particles":

            l_output += l_tabString + "index_t l_index = get_global_id(0);"
            l_output += "\n"
        
        l_output += "\n"
        l_output += l_tabString + "//a_standardValues[0].frame;"
        l_output += "\n"
        l_output += l_tabString + "//a_standardValues[0].time;"
        l_output += "\n"
        l_output += l_tabString + "//a_standardValues[0].realTimeStep; Without time scale."
        l_output += "\n"
        l_output += l_tabString + "//a_standardValues[0].solverTimeStep; With time scale."
        l_output += "\n"
        l_output += l_tabString + "//a_standardValues[0].timeScale;"
        l_output += "\n"
        l_output += l_tabString + "//a_standardValues[0].subSteps;"
        l_output += "\n"

        l_output += "\n"
        l_output += l_tabString + "//float l_fieldValue = coreReadFieldValue(l_index, a_fieldArgument);"
        l_output += "\n"
        l_output += "\n"
        l_output += l_tabString + "//coreWriteFieldValue(l_fieldValue, l_index, a_fieldArgument);"
        l_output += "\n"

        l_output += "\n"
        l_output += "}"

        l_codeParameter.set(l_output)
        

# =====================================================================
# Section 4 - OpenCL kernel call sites in fluidSim.cl (enqueue target list)
# =====================================================================

FLUIDSIM_CL_KERNELS = [
    ("coreClearScalarInt32", 5475),  # __kernel void coreClearScalarInt32(
    ("coreClearCoupleInt32", 5487),  # __kernel void coreClearCoupleInt32(
    ("coreClearScalarInt16", 5501),  # __kernel void coreClearScalarInt16(
    ("coreClearScalarInt8", 5513),  # __kernel void coreClearScalarInt8(
    ("coreClearScalarVoxelType", 5525),  # __kernel void coreClearScalarVoxelType(
    ("coreClearScalarFloat32", 5537),  # __kernel void coreClearScalarFloat32(
    ("coreClearScalarField", 5549),  # __kernel void coreClearScalarField(
    ("coreClearVectorField", 5561),  # __kernel void coreClearVectorField(
    ("coreComputeMaxActiveValue", 5579),  # __kernel void coreComputeMaxActiveValue(
    ("coreConvertFieldVectorToMagnitude", 5624),  # __kernel void coreConvertFieldVectorToMagnitude(
    ("coreConvertFieldStaggeredVectorToMagnitude", 5647),  # __kernel void coreConvertFieldStaggeredVectorToMagnitude(
    ("coreResetScaleField", 5686),  # __kernel void coreResetScaleField(
    ("coreRampToScaleFieldVolume", 5700),  # __kernel void coreRampToScaleFieldVolume(
    ("coreRampToScaleFieldParticle", 5764),  # __kernel void coreRampToScaleFieldParticle(
    ("coreSyncScalarInt32", 5822),  # __kernel void coreSyncScalarInt32(
    ("coreSyncScalarInt16", 5835),  # __kernel void coreSyncScalarInt16(
    ("coreSyncScalarVoxelType", 5848),  # __kernel void coreSyncScalarVoxelType(
    ("coreSyncScalarFloat32", 5861),  # __kernel void coreSyncScalarFloat32(
    ("coreSyncScalarFloat16", 5874),  # __kernel void coreSyncScalarFloat16(
    ("coreSyncScalarField", 5889),  # __kernel void coreSyncScalarField(
    ("coreSyncCoupleInt32", 5902),  # __kernel void coreSyncCoupleInt32(
    ("coreSyncVectorFloat32", 5921),  # __kernel void coreSyncVectorFloat32(
    ("coreSyncCoupleField", 5944),  # __kernel void coreSyncCoupleField(
    ("coreSyncVectorField", 5963),  # __kernel void coreSyncVectorField(
    ("coreReduceScalarFieldDotPartialLinear128", 5988),  # __kernel void coreReduceScalarFieldDotPartialLinear128(
    ("coreReduceScalarFieldDotPartialLinear256", 6066),  # __kernel void coreReduceScalarFieldDotPartialLinear256(
    ("coreReduceScalarFieldDotPartialLinear512", 6144),  # __kernel void coreReduceScalarFieldDotPartialLinear512(
    ("coreReduceScalarFieldDotPartialLinear1024", 6222),  # __kernel void coreReduceScalarFieldDotPartialLinear1024(
    ("coreReduceVectorFieldDotPartialLinear128", 6301),  # __kernel void coreReduceVectorFieldDotPartialLinear128(
    ("coreReduceVectorFieldDotPartialLinear256", 6399),  # __kernel void coreReduceVectorFieldDotPartialLinear256(
    ("coreReduceVectorFieldDotPartialLinear512", 6497),  # __kernel void coreReduceVectorFieldDotPartialLinear512(
    ("coreReduceVectorFieldDotPartialLinear1024", 6595),  # __kernel void coreReduceVectorFieldDotPartialLinear1024(
    ("coreReduceSum128", 6694),  # __kernel void coreReduceSum128(
    ("coreReduceSum256", 6763),  # __kernel void coreReduceSum256(
    ("coreReduceSum512", 6832),  # __kernel void coreReduceSum512(
    ("coreReduceSum1024", 6901),  # __kernel void coreReduceSum1024(
    ("coreCreateSDFAxisX", 6973),  # __kernel void coreCreateSDFAxisX(
    ("coreCreateSDFAxisY", 7041),  # __kernel void coreCreateSDFAxisY(
    ("coreCreateSDFAxisZ", 7128),  # __kernel void coreCreateSDFAxisZ(
    ("coreSmoothSDF", 7219),  # __kernel void coreSmoothSDF(
    ("coreRenormalizeSdfFastSweep", 7312),  # __kernel void coreRenormalizeSdfFastSweep(
    ("coreRecomputeCollisionBitFromSdf", 7447),  # __kernel void coreRecomputeCollisionBitFromSdf(
    ("coreSDFToGradient", 7474),  # __kernel void coreSDFToGradient(
    ("coreConvertFloat32ToFloat16", 7596),  # __kernel void coreConvertFloat32ToFloat16(
    ("coreConvertFloat16ToFloat32", 7617),  # __kernel void coreConvertFloat16ToFloat32(
    ("coreConvertInt32ToInt16", 7638),  # __kernel void coreConvertInt32ToInt16(
    ("coreConvertInt16ToInt32", 7651),  # __kernel void coreConvertInt16ToInt32(
    ("coreClampScalarField", 7666),  # __kernel void coreClampScalarField(
    ("coreRemapScalarField", 7685),  # __kernel void coreRemapScalarField(
    ("coreSortScalarInt32", 7704),  # __kernel void coreSortScalarInt32(
    ("coreSortScalarInt16", 7718),  # __kernel void coreSortScalarInt16(
    ("coreSortScalarVoxelType", 7732),  # __kernel void coreSortScalarVoxelType(
    ("coreSortScalarFloat16", 7746),  # __kernel void coreSortScalarFloat16(
    ("coreSortScalarFloat32", 7762),  # __kernel void coreSortScalarFloat32(
    ("coreSortVectorFloat32", 7776),  # __kernel void coreSortVectorFloat32(
    ("coreSortScalarField", 7802),  # __kernel void coreSortScalarField(
    ("coreSortCoupleField", 7816),  # __kernel void coreSortCoupleField(
    ("coreSortVectorField", 7838),  # __kernel void coreSortVectorField(
    ("coreGridToParticleFieldAttribute", 7866),  # __kernel void coreGridToParticleFieldAttribute(
    ("coreParticleToGridCollect", 7925),  # __kernel void coreParticleToGridCollect(
    ("coreGridExpand", 8012),  # __kernel void coreGridExpand(
    ("coreParticleToGridAverage", 8072),  # __kernel void coreParticleToGridAverage(
    ("coreSourceNanoVDBTypeFog", 8130),  # __kernel void coreSourceNanoVDBTypeFog(
    ("coreSourceNanoVDBTypeFogWithTemperature", 8239),  # __kernel void coreSourceNanoVDBTypeFogWithTemperature(
    ("coreSourceNanoVDBTypeSDF", 8351),  # __kernel void coreSourceNanoVDBTypeSDF(
    ("coreSourceNanoVDBTypeSDFWithTemperature", 8460),  # __kernel void coreSourceNanoVDBTypeSDFWithTemperature(
    ("coreSourceNanoVDBMaskFog", 8572),  # __kernel void coreSourceNanoVDBMaskFog(
    ("coreSourceNanoVDBMaskSDF", 8663),  # __kernel void coreSourceNanoVDBMaskSDF(
    ("coreSourceNanoVDBFloat", 8754),  # __kernel void coreSourceNanoVDBFloat(
    ("coreSourceNanoVDBVector", 8856),  # __kernel void coreSourceNanoVDBVector(
    ("coreSourceOpenVDBMask", 8955),  # __kernel void coreSourceOpenVDBMask(
    ("coreSourceOpenVDBType", 8989),  # __kernel void coreSourceOpenVDBType(
    ("coreSourceOpenVDBTypeWithTemperature", 9037),  # __kernel void coreSourceOpenVDBTypeWithTemperature(
    ("coreSourceOpenVDBFloat", 9088),  # __kernel void coreSourceOpenVDBFloat(
    ("coreSourceOpenVDBVectorComponent", 9129),  # __kernel void coreSourceOpenVDBVectorComponent(
    ("coreSourceFloat", 9155),  # __kernel void coreSourceFloat(
    ("coreSourceVector", 9210),  # __kernel void coreSourceVector(
    ("coreSourceStaggeredVector", 9306),  # __kernel void coreSourceStaggeredVector(
    ("coreSourceFloatParticleAttribute", 9405),  # __kernel void coreSourceFloatParticleAttribute(
    ("coreSourceVectorParticleAttribute", 9503),  # __kernel void coreSourceVectorParticleAttribute(
    ("paradigmParticleToVoxelCountAndUpdateVoxelIndex", 9645),  # __kernel void paradigmParticleToVoxelCountAndUpdateVoxelIndex(
    ("paradigmVoxelCountToVoxelParticleArrayStart", 9714),  # __kernel void paradigmVoxelCountToVoxelParticleArrayStart(
    ("paradigmParticleToVoxelParticleArray", 9758),  # __kernel void paradigmParticleToVoxelParticleArray(
    ("paradigmVoxelParticleArraySort", 9803),  # __kernel void paradigmVoxelParticleArraySort(
    ("paradigmAdvectParticles", 10243),  # __kernel void paradigmAdvectParticles(
    ("paradigmAdvectWhiteWater", 10690),  # __kernel void paradigmAdvectWhiteWater(
    ("paradigmSourceShapes", 11184),  # __kernel void paradigmSourceShapes(
    ("paradigmSourceShapesParticlesAttributes", 11849),  # __kernel void paradigmSourceShapesParticlesAttributes(
    ("paradigmSourceHeight", 12293),  # __kernel void paradigmSourceHeight(
    ("paradigmSourceResolveType", 12385),  # __kernel void paradigmSourceResolveType(
    ("paradigmSourceParticles", 12413),  # __kernel void paradigmSourceParticles(
    ("paradigmSourceParticleVectorFieldAttribute", 12482),  # __kernel void paradigmSourceParticleVectorFieldAttribute(
    ("paradigmSourceParticleScalarFieldAttribute", 12516),  # __kernel void paradigmSourceParticleScalarFieldAttribute(
    ("paradigmSourceParticleScalarIntegerAttribute", 12540),  # __kernel void paradigmSourceParticleScalarIntegerAttribute(
    ("paradigmSourceParticleScalarFloatAttribute", 12562),  # __kernel void paradigmSourceParticleScalarFloatAttribute(
    ("paradigmSourceParticleScalarHalfAttribute", 12586),  # __kernel void paradigmSourceParticleScalarHalfAttribute(
    ("paradigmForcesGravity", 12614),  # __kernel void paradigmForcesGravity(
    ("paradigmForcesDirectionalForce", 12643),  # __kernel void paradigmForcesDirectionalForce(
    ("paradigmForcesOmnidirectionalForce", 12669),  # __kernel void paradigmForcesOmnidirectionalForce(
    ("paradigmForcesSpeedLimit", 12706),  # __kernel void paradigmForcesSpeedLimit(
    ("paradigmForcesDiffusion", 12748),  # __kernel void paradigmForcesDiffusion(
    ("paradigmForcesDissipateParticleAttribute", 12855),  # __kernel void paradigmForcesDissipateParticleAttribute(
    ("paradigmAttractMaskToField", 12883),  # __kernel void paradigmAttractMaskToField(
    ("paradigmAttractSmoothField", 12906),  # __kernel void paradigmAttractSmoothField(
    ("paradigmAttractDirection", 12981),  # __kernel void paradigmAttractDirection(
    ("paradigmAttractSdfGradientDirection", 13101),  # __kernel void paradigmAttractSdfGradientDirection(
    ("paradigmProjectNonDivergentCollisionPreConditionerNew", 13262),  # __kernel void paradigmProjectNonDivergentCollisionPreConditionerNew(
    ("paradigmProjectNonDivergentComputeDivergence_old", 13310),  # __kernel void paradigmProjectNonDivergentComputeDivergence_old(
    ("paradigmProjectNonDivergentComputeDivergence", 13475),  # __kernel void paradigmProjectNonDivergentComputeDivergence(
    ("paradigmProjectNonDivergentElectrostaticBuildLeafAggregates8", 13676),  # __kernel void paradigmProjectNonDivergentElectrostaticBuildLeafAggregates8(
    ("paradigmProjectNonDivergentElectrostaticBuildLeafAggregates16", 13760),  # __kernel void paradigmProjectNonDivergentElectrostaticBuildLeafAggregates16(
    ("paradigmProjectNonDivergentElectrostaticBuildLeafAggregates32", 13844),  # __kernel void paradigmProjectNonDivergentElectrostaticBuildLeafAggregates32(
    ("paradigmProjectNonDivergentElectrostaticSolvePotential8", 13929),  # __kernel void paradigmProjectNonDivergentElectrostaticSolvePotential8(
    ("paradigmProjectNonDivergentElectrostaticSolvePotential16", 14114),  # __kernel void paradigmProjectNonDivergentElectrostaticSolvePotential16(
    ("paradigmProjectNonDivergentElectrostaticSolvePotential32", 14299),  # __kernel void paradigmProjectNonDivergentElectrostaticSolvePotential32(
    ("paradigmProjectNonDivergentProject", 14485),  # __kernel void paradigmProjectNonDivergentProject(
    ("paradigmProjectNonDivergentExpandVelocityIntoCollision", 14601),  # __kernel void paradigmProjectNonDivergentExpandVelocityIntoCollision(
    ("paradigmProjectNonDivergentJacobi2_original", 14730),  # __kernel void paradigmProjectNonDivergentJacobi2_original(
    ("paradigmProjectNonDivergentJacobi4_original", 14840),  # __kernel void paradigmProjectNonDivergentJacobi4_original(
    ("paradigmProjectNonDivergentJacobi8_original", 14950),  # __kernel void paradigmProjectNonDivergentJacobi8_original(
    ("paradigmProjectNonDivergentJacobi16_original", 15060),  # __kernel void paradigmProjectNonDivergentJacobi16_original(
    ("paradigmProjectNonDivergentJacobi32_original", 15170),  # __kernel void paradigmProjectNonDivergentJacobi32_original(
    ("paradigmProjectNonDivergentJacobi2_optimizedUnroll", 15281),  # __kernel void paradigmProjectNonDivergentJacobi2_optimizedUnroll(
    ("paradigmProjectNonDivergentJacobi4_optimizedUnroll", 15529),  # __kernel void paradigmProjectNonDivergentJacobi4_optimizedUnroll(
    ("paradigmProjectNonDivergentJacobi8_optimizedUnroll", 15777),  # __kernel void paradigmProjectNonDivergentJacobi8_optimizedUnroll(
    ("paradigmProjectNonDivergentJacobi16_optimizedUnroll", 16025),  # __kernel void paradigmProjectNonDivergentJacobi16_optimizedUnroll(
    ("paradigmProjectNonDivergentJacobi32_optimizedUnroll", 16273),  # __kernel void paradigmProjectNonDivergentJacobi32_optimizedUnroll(
    ("paradigmProjectNonDivergentSampleDown", 16522),  # __kernel void paradigmProjectNonDivergentSampleDown(
    ("paradigmProjectNonDivergentSampleUp", 16594),  # __kernel void paradigmProjectNonDivergentSampleUp(
    ("paradigmProjectNonDivergentComputeInverseDiagonal8", 16665),  # __kernel void paradigmProjectNonDivergentComputeInverseDiagonal8(
    ("paradigmProjectNonDivergentComputeInverseDiagonal16", 16748),  # __kernel void paradigmProjectNonDivergentComputeInverseDiagonal16(
    ("paradigmProjectNonDivergentComputeInverseDiagonal32", 16831),  # __kernel void paradigmProjectNonDivergentComputeInverseDiagonal32(
    ("paradigmProjectNonDivergentInitializePcg", 16915),  # __kernel void paradigmProjectNonDivergentInitializePcg(
    ("paradigmProjectNonDivergentApplyMatrix8", 16955),  # __kernel void paradigmProjectNonDivergentApplyMatrix8(
    ("paradigmProjectNonDivergentApplyMatrix16", 17125),  # __kernel void paradigmProjectNonDivergentApplyMatrix16(
    ("paradigmProjectNonDivergentApplyMatrix32", 17295),  # __kernel void paradigmProjectNonDivergentApplyMatrix32(
    ("paradigmProjectNonDivergentComputeAlpha", 17466),  # __kernel void paradigmProjectNonDivergentComputeAlpha(
    ("paradigmProjectNonDivergentUpdateXR", 17484),  # __kernel void paradigmProjectNonDivergentUpdateXR(
    ("paradigmProjectNonDivergentApplyPreConditioner", 17511),  # __kernel void paradigmProjectNonDivergentApplyPreConditioner(
    ("paradigmProjectNonDivergentComputeBeta", 17542),  # __kernel void paradigmProjectNonDivergentComputeBeta(
    ("paradigmProjectNonDivergentUpdateP", 17562),  # __kernel void paradigmProjectNonDivergentUpdateP(
    ("paradigmComputeMaxActiveValue", 17593),  # __kernel void paradigmComputeMaxActiveValue(
    ("paradigmParticlesRelax", 17646),  # __kernel void paradigmParticlesRelax(
    ("paradigmUpdateParticleVoxelIndex", 17768),  # __kernel void paradigmUpdateParticleVoxelIndex(
    ("paradigmClearJustInitializedVoxelTypeTracked", 17793),  # __kernel void paradigmClearJustInitializedVoxelTypeTracked(
    ("paradigmLeafSortVoxelTypeTracked", 17821),  # __kernel void paradigmLeafSortVoxelTypeTracked(
    ("paradigmResetSourceShapeTrackedVoxelType", 17855),  # __kernel void paradigmResetSourceShapeTrackedVoxelType(
    ("paradigmParticleAge", 17874),  # __kernel void paradigmParticleAge(
    ("paradigmParticleLifespan", 17893),  # __kernel void paradigmParticleLifespan(
    ("paradigmComputeChurn", 17918),  # __kernel void paradigmComputeChurn(
    ("paradigmExpandFluidMask", 18047),  # __kernel void paradigmExpandFluidMask(
    ("paradigmEmitWhiteWater", 18107),  # __kernel void paradigmEmitWhiteWater(
    ("paradigmParticleInSinkCheck", 18271),  # __kernel void paradigmParticleInSinkCheck(
    ("paradigmSortParticlesRandomMutex", 18334),  # __kernel void paradigmSortParticlesRandomMutex(
    ("paradigmParticleToLeafCount", 18366),  # __kernel void paradigmParticleToLeafCount(
    ("paradigmParticleLeafCountToLeafArrayStart", 18411),  # __kernel void paradigmParticleLeafCountToLeafArrayStart(
    ("paradigmSortParticlesLeafArrayMutex", 18455),  # __kernel void paradigmSortParticlesLeafArrayMutex(
    ("paradigmParticleSetGridData", 18530),  # __kernel void paradigmParticleSetGridData(
    ("paradigmWhiteWaterSetGridData", 18572),  # __kernel void paradigmWhiteWaterSetGridData(
    ("paradigmParticleToCount", 18608),  # __kernel void paradigmParticleToCount(
    ("paradigmParticleToCountPressure", 18675),  # __kernel void paradigmParticleToCountPressure(
    ("paradigmParticleCountToPressure", 18750),  # __kernel void paradigmParticleCountToPressure(
    ("paradigmPressureThresholdCompute", 18796),  # __kernel void paradigmPressureThresholdCompute(
    ("paradigmPressureThresholdApply", 18816),  # __kernel void paradigmPressureThresholdApply(
    ("paradigmPressureScale", 18833),  # __kernel void paradigmPressureScale(
    ("paradigmStaggeredVectorFieldToParticle", 18900),  # __kernel void paradigmStaggeredVectorFieldToParticle(
    ("paradigmParticleToGridActive", 18987),  # __kernel void paradigmParticleToGridActive(
    ("paradigmBuildSurfaceSdf", 19068),  # __kernel void paradigmBuildSurfaceSdf(
    ("paradigmSmoothSurfaceSdfDialate", 19174),  # __kernel void paradigmSmoothSurfaceSdfDialate(
    ("paradigmSmoothSurfaceSdfErode", 19189),  # __kernel void paradigmSmoothSurfaceSdfErode(
    ("paradigmSmoothSurfaceSdfBox", 19204),  # __kernel void paradigmSmoothSurfaceSdfBox(
    ("paradigmSmoothSurfaceSdfMeanCurvatureFlow", 19265),  # __kernel void paradigmSmoothSurfaceSdfMeanCurvatureFlow(
    ("paradigmForceViscositySmooth", 19448),  # __kernel void paradigmForceViscositySmooth(
    ("paradigmForceViscosityCounterIncrement", 19747),  # __kernel void paradigmForceViscosityCounterIncrement(
    ("paradigmForceViscosityBuildRhs", 19759),  # __kernel void paradigmForceViscosityBuildRhs(
    ("paradigmForceViscosityComputeInverseDiagonal", 19923),  # __kernel void paradigmForceViscosityComputeInverseDiagonal(
    ("paradigmForceViscosityApplyMatrix", 20016),  # __kernel void paradigmForceViscosityApplyMatrix(
    ("paradigmForceViscosityInitializePcg", 20199),  # __kernel void paradigmForceViscosityInitializePcg(
    ("paradigmForceViscosityFinalizeInitialResidual", 20226),  # __kernel void paradigmForceViscosityFinalizeInitialResidual(
    ("paradigmForceViscosityInitializeDirectionFromPreconditioned", 20265),  # __kernel void paradigmForceViscosityInitializeDirectionFromPreconditioned(
    ("paradigmForceViscosityComputeAlpha", 20288),  # __kernel void paradigmForceViscosityComputeAlpha(
    ("paradigmForceViscosityUpdateXR", 20307),  # __kernel void paradigmForceViscosityUpdateXR(
    ("paradigmForceViscosityApplyPreConditioner", 20339),  # __kernel void paradigmForceViscosityApplyPreConditioner(
    ("paradigmForceViscosityComputeBeta", 20368),  # __kernel void paradigmForceViscosityComputeBeta(
    ("paradigmForceViscosityUpdateP", 20389),  # __kernel void paradigmForceViscosityUpdateP(
    ("paradigmForceViscosityApplyCollision", 20419),  # __kernel void paradigmForceViscosityApplyCollision(
    ("paradigmSdfToAxisCollision", 20460),  # __kernel void paradigmSdfToAxisCollision(
    ("paradigmExpandAndShiftAxisVelocityInColliderAxis", 20537),  # __kernel void paradigmExpandAndShiftAxisVelocityInColliderAxis(
]