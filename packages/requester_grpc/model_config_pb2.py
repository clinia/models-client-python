# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: model_config.proto
# Protobuf Python Version: 4.25.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12model_config.proto\x12\tinference\"\xc0\x01\n\x10ModelRateLimiter\x12\x42\n\tresources\x18\x01 \x03(\x0b\x32$.inference.ModelRateLimiter.ResourceR\tresources\x12\x1a\n\x08priority\x18\x02 \x01(\rR\x08priority\x1aL\n\x08Resource\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x16\n\x06global\x18\x02 \x01(\x08R\x06global\x12\x14\n\x05\x63ount\x18\x03 \x01(\rR\x05\x63ount\"\xed\x04\n\x12ModelInstanceGroup\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x36\n\x04kind\x18\x04 \x01(\x0e\x32\".inference.ModelInstanceGroup.KindR\x04kind\x12\x14\n\x05\x63ount\x18\x02 \x01(\x05R\x05\x63ount\x12>\n\x0crate_limiter\x18\x06 \x01(\x0b\x32\x1b.inference.ModelRateLimiterR\x0brateLimiter\x12\x12\n\x04gpus\x18\x03 \x03(\x05R\x04gpus\x12Z\n\x11secondary_devices\x18\x08 \x03(\x0b\x32-.inference.ModelInstanceGroup.SecondaryDeviceR\x10secondaryDevices\x12\x18\n\x07profile\x18\x05 \x03(\tR\x07profile\x12\x18\n\x07passive\x18\x07 \x01(\x08R\x07passive\x12\x1f\n\x0bhost_policy\x18\t \x01(\tR\nhostPolicy\x1a\xac\x01\n\x0fSecondaryDevice\x12U\n\x04kind\x18\x01 \x01(\x0e\x32\x41.inference.ModelInstanceGroup.SecondaryDevice.SecondaryDeviceKindR\x04kind\x12\x1b\n\tdevice_id\x18\x02 \x01(\x03R\x08\x64\x65viceId\"%\n\x13SecondaryDeviceKind\x12\x0e\n\nKIND_NVDLA\x10\x00\"A\n\x04Kind\x12\r\n\tKIND_AUTO\x10\x00\x12\x0c\n\x08KIND_GPU\x10\x01\x12\x0c\n\x08KIND_CPU\x10\x02\x12\x0e\n\nKIND_MODEL\x10\x03\"*\n\x12ModelTensorReshape\x12\x14\n\x05shape\x18\x01 \x03(\x03R\x05shape\"\x84\x03\n\nModelInput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x30\n\tdata_type\x18\x02 \x01(\x0e\x32\x13.inference.DataTypeR\x08\x64\x61taType\x12\x34\n\x06\x66ormat\x18\x03 \x01(\x0e\x32\x1c.inference.ModelInput.FormatR\x06\x66ormat\x12\x12\n\x04\x64ims\x18\x04 \x03(\x03R\x04\x64ims\x12\x37\n\x07reshape\x18\x05 \x01(\x0b\x32\x1d.inference.ModelTensorReshapeR\x07reshape\x12&\n\x0fis_shape_tensor\x18\x06 \x01(\x08R\risShapeTensor\x12,\n\x12\x61llow_ragged_batch\x18\x07 \x01(\x08R\x10\x61llowRaggedBatch\x12\x1a\n\x08optional\x18\x08 \x01(\x08R\x08optional\";\n\x06\x46ormat\x12\x0f\n\x0b\x46ORMAT_NONE\x10\x00\x12\x0f\n\x0b\x46ORMAT_NHWC\x10\x01\x12\x0f\n\x0b\x46ORMAT_NCHW\x10\x02\"\xef\x01\n\x0bModelOutput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x30\n\tdata_type\x18\x02 \x01(\x0e\x32\x13.inference.DataTypeR\x08\x64\x61taType\x12\x12\n\x04\x64ims\x18\x03 \x03(\x03R\x04\x64ims\x12\x37\n\x07reshape\x18\x05 \x01(\x0b\x32\x1d.inference.ModelTensorReshapeR\x07reshape\x12%\n\x0elabel_filename\x18\x04 \x01(\tR\rlabelFilename\x12&\n\x0fis_shape_tensor\x18\x06 \x01(\x08R\risShapeTensor\"\x82\x03\n\nBatchInput\x12.\n\x04kind\x18\x01 \x01(\x0e\x32\x1a.inference.BatchInput.KindR\x04kind\x12\x1f\n\x0btarget_name\x18\x02 \x03(\tR\ntargetName\x12\x30\n\tdata_type\x18\x03 \x01(\x0e\x32\x13.inference.DataTypeR\x08\x64\x61taType\x12!\n\x0csource_input\x18\x04 \x03(\tR\x0bsourceInput\"\xcd\x01\n\x04Kind\x12\x17\n\x13\x42\x41TCH_ELEMENT_COUNT\x10\x00\x12#\n\x1f\x42\x41TCH_ACCUMULATED_ELEMENT_COUNT\x10\x01\x12-\n)BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO\x10\x02\x12$\n BATCH_MAX_ELEMENT_COUNT_AS_SHAPE\x10\x03\x12\x14\n\x10\x42\x41TCH_ITEM_SHAPE\x10\x04\x12\x1c\n\x18\x42\x41TCH_ITEM_SHAPE_FLATTEN\x10\x05\"\xae\x01\n\x0b\x42\x61tchOutput\x12\x1f\n\x0btarget_name\x18\x01 \x03(\tR\ntargetName\x12/\n\x04kind\x18\x02 \x01(\x0e\x32\x1b.inference.BatchOutput.KindR\x04kind\x12!\n\x0csource_input\x18\x03 \x03(\tR\x0bsourceInput\"*\n\x04Kind\x12\"\n\x1e\x42\x41TCH_SCATTER_WITH_INPUT_SHAPE\x10\x00\"\xbe\x02\n\x12ModelVersionPolicy\x12>\n\x06latest\x18\x01 \x01(\x0b\x32$.inference.ModelVersionPolicy.LatestH\x00R\x06latest\x12\x35\n\x03\x61ll\x18\x02 \x01(\x0b\x32!.inference.ModelVersionPolicy.AllH\x00R\x03\x61ll\x12\x44\n\x08specific\x18\x03 \x01(\x0b\x32&.inference.ModelVersionPolicy.SpecificH\x00R\x08specific\x1a+\n\x06Latest\x12!\n\x0cnum_versions\x18\x01 \x01(\rR\x0bnumVersions\x1a\x05\n\x03\x41ll\x1a&\n\x08Specific\x12\x1a\n\x08versions\x18\x01 \x03(\x03R\x08versionsB\x0f\n\rpolicy_choice\"\xe6\x10\n\x17ModelOptimizationPolicy\x12>\n\x05graph\x18\x01 \x01(\x0b\x32(.inference.ModelOptimizationPolicy.GraphR\x05graph\x12L\n\x08priority\x18\x02 \x01(\x0e\x32\x30.inference.ModelOptimizationPolicy.ModelPriorityR\x08priority\x12;\n\x04\x63uda\x18\x03 \x01(\x0b\x32\'.inference.ModelOptimizationPolicy.CudaR\x04\x63uda\x12o\n\x16\x65xecution_accelerators\x18\x04 \x01(\x0b\x32\x38.inference.ModelOptimizationPolicy.ExecutionAcceleratorsR\x15\x65xecutionAccelerators\x12\x65\n\x13input_pinned_memory\x18\x05 \x01(\x0b\x32\x35.inference.ModelOptimizationPolicy.PinnedMemoryBufferR\x11inputPinnedMemory\x12g\n\x14output_pinned_memory\x18\x06 \x01(\x0b\x32\x35.inference.ModelOptimizationPolicy.PinnedMemoryBufferR\x12outputPinnedMemory\x12\x43\n\x1egather_kernel_buffer_threshold\x18\x07 \x01(\rR\x1bgatherKernelBufferThreshold\x12%\n\x0e\x65\x61ger_batching\x18\x08 \x01(\x08R\reagerBatching\x1a\x1d\n\x05Graph\x12\x14\n\x05level\x18\x01 \x01(\x05R\x05level\x1a\xc1\x06\n\x04\x43uda\x12\x16\n\x06graphs\x18\x01 \x01(\x08R\x06graphs\x12(\n\x10\x62usy_wait_events\x18\x02 \x01(\x08R\x0e\x62usyWaitEvents\x12P\n\ngraph_spec\x18\x03 \x03(\x0b\x32\x31.inference.ModelOptimizationPolicy.Cuda.GraphSpecR\tgraphSpec\x12,\n\x12output_copy_stream\x18\x04 \x01(\x08R\x10outputCopyStream\x1a\xf6\x04\n\tGraphSpec\x12\x1d\n\nbatch_size\x18\x01 \x01(\x05R\tbatchSize\x12R\n\x05input\x18\x02 \x03(\x0b\x32<.inference.ModelOptimizationPolicy.Cuda.GraphSpec.InputEntryR\x05input\x12h\n\x11graph_lower_bound\x18\x03 \x01(\x0b\x32<.inference.ModelOptimizationPolicy.Cuda.GraphSpec.LowerBoundR\x0fgraphLowerBound\x1a\x19\n\x05Shape\x12\x10\n\x03\x64im\x18\x01 \x03(\x03R\x03\x64im\x1a\xfd\x01\n\nLowerBound\x12\x1d\n\nbatch_size\x18\x01 \x01(\x05R\tbatchSize\x12]\n\x05input\x18\x02 \x03(\x0b\x32G.inference.ModelOptimizationPolicy.Cuda.GraphSpec.LowerBound.InputEntryR\x05input\x1aq\n\nInputEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12M\n\x05value\x18\x02 \x01(\x0b\x32\x37.inference.ModelOptimizationPolicy.Cuda.GraphSpec.ShapeR\x05value:\x02\x38\x01\x1aq\n\nInputEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12M\n\x05value\x18\x02 \x01(\x0b\x32\x37.inference.ModelOptimizationPolicy.Cuda.GraphSpec.ShapeR\x05value:\x02\x38\x01\x1a\xf6\x03\n\x15\x45xecutionAccelerators\x12\x80\x01\n\x19gpu_execution_accelerator\x18\x01 \x03(\x0b\x32\x44.inference.ModelOptimizationPolicy.ExecutionAccelerators.AcceleratorR\x17gpuExecutionAccelerator\x12\x80\x01\n\x19\x63pu_execution_accelerator\x18\x02 \x03(\x0b\x32\x44.inference.ModelOptimizationPolicy.ExecutionAccelerators.AcceleratorR\x17\x63puExecutionAccelerator\x1a\xd6\x01\n\x0b\x41\x63\x63\x65lerator\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12t\n\nparameters\x18\x02 \x03(\x0b\x32T.inference.ModelOptimizationPolicy.ExecutionAccelerators.Accelerator.ParametersEntryR\nparameters\x1a=\n\x0fParametersEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\x1a,\n\x12PinnedMemoryBuffer\x12\x16\n\x06\x65nable\x18\x01 \x01(\x08R\x06\x65nable\"I\n\rModelPriority\x12\x14\n\x10PRIORITY_DEFAULT\x10\x00\x12\x10\n\x0cPRIORITY_MAX\x10\x01\x12\x10\n\x0cPRIORITY_MIN\x10\x02\"\xaa\x02\n\x10ModelQueuePolicy\x12P\n\x0etimeout_action\x18\x01 \x01(\x0e\x32).inference.ModelQueuePolicy.TimeoutActionR\rtimeoutAction\x12@\n\x1c\x64\x65\x66\x61ult_timeout_microseconds\x18\x02 \x01(\x04R\x1a\x64\x65\x66\x61ultTimeoutMicroseconds\x12\x34\n\x16\x61llow_timeout_override\x18\x03 \x01(\x08R\x14\x61llowTimeoutOverride\x12$\n\x0emax_queue_size\x18\x04 \x01(\rR\x0cmaxQueueSize\"&\n\rTimeoutAction\x12\n\n\x06REJECT\x10\x00\x12\t\n\x05\x44\x45LAY\x10\x01\"\xb7\x04\n\x14ModelDynamicBatching\x12\x30\n\x14preferred_batch_size\x18\x01 \x03(\x05R\x12preferredBatchSize\x12?\n\x1cmax_queue_delay_microseconds\x18\x02 \x01(\x04R\x19maxQueueDelayMicroseconds\x12+\n\x11preserve_ordering\x18\x03 \x01(\x08R\x10preserveOrdering\x12\'\n\x0fpriority_levels\x18\x04 \x01(\x04R\x0epriorityLevels\x12\x34\n\x16\x64\x65\x66\x61ult_priority_level\x18\x05 \x01(\x04R\x14\x64\x65\x66\x61ultPriorityLevel\x12M\n\x14\x64\x65\x66\x61ult_queue_policy\x18\x06 \x01(\x0b\x32\x1b.inference.ModelQueuePolicyR\x12\x64\x65\x66\x61ultQueuePolicy\x12l\n\x15priority_queue_policy\x18\x07 \x03(\x0b\x32\x38.inference.ModelDynamicBatching.PriorityQueuePolicyEntryR\x13priorityQueuePolicy\x1a\x63\n\x18PriorityQueuePolicyEntry\x12\x10\n\x03key\x18\x01 \x01(\x04R\x03key\x12\x31\n\x05value\x18\x02 \x01(\x0b\x32\x1b.inference.ModelQueuePolicyR\x05value:\x02\x38\x01\"\xab\x0e\n\x15ModelSequenceBatching\x12I\n\x06\x64irect\x18\x03 \x01(\x0b\x32/.inference.ModelSequenceBatching.StrategyDirectH\x00R\x06\x64irect\x12I\n\x06oldest\x18\x04 \x01(\x0b\x32/.inference.ModelSequenceBatching.StrategyOldestH\x00R\x06oldest\x12\x43\n\x1emax_sequence_idle_microseconds\x18\x01 \x01(\x04R\x1bmaxSequenceIdleMicroseconds\x12R\n\rcontrol_input\x18\x02 \x03(\x0b\x32-.inference.ModelSequenceBatching.ControlInputR\x0c\x63ontrolInput\x12<\n\x05state\x18\x05 \x03(\x0b\x32&.inference.ModelSequenceBatching.StateR\x05state\x12-\n\x12iterative_sequence\x18\x06 \x01(\x08R\x11iterativeSequence\x1a\xef\x02\n\x07\x43ontrol\x12\x41\n\x04kind\x18\x01 \x01(\x0e\x32-.inference.ModelSequenceBatching.Control.KindR\x04kind\x12(\n\x10int32_false_true\x18\x02 \x03(\x05R\x0eint32FalseTrue\x12&\n\x0f\x66p32_false_true\x18\x03 \x03(\x02R\rfp32FalseTrue\x12&\n\x0f\x62ool_false_true\x18\x05 \x03(\x08R\rboolFalseTrue\x12\x30\n\tdata_type\x18\x04 \x01(\x0e\x32\x13.inference.DataTypeR\x08\x64\x61taType\"u\n\x04Kind\x12\x1a\n\x16\x43ONTROL_SEQUENCE_START\x10\x00\x12\x1a\n\x16\x43ONTROL_SEQUENCE_READY\x10\x01\x12\x18\n\x14\x43ONTROL_SEQUENCE_END\x10\x02\x12\x1b\n\x17\x43ONTROL_SEQUENCE_CORRID\x10\x03\x1a\x66\n\x0c\x43ontrolInput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x42\n\x07\x63ontrol\x18\x02 \x03(\x0b\x32(.inference.ModelSequenceBatching.ControlR\x07\x63ontrol\x1a\xb4\x01\n\x0cInitialState\x12\x30\n\tdata_type\x18\x01 \x01(\x0e\x32\x13.inference.DataTypeR\x08\x64\x61taType\x12\x12\n\x04\x64ims\x18\x02 \x03(\x03R\x04\x64ims\x12\x1d\n\tzero_data\x18\x03 \x01(\x08H\x00R\x08zeroData\x12\x1d\n\tdata_file\x18\x04 \x01(\tH\x00R\x08\x64\x61taFile\x12\x12\n\x04name\x18\x05 \x01(\tR\x04nameB\x0c\n\nstate_data\x1a\xd8\x02\n\x05State\x12\x1d\n\ninput_name\x18\x01 \x01(\tR\tinputName\x12\x1f\n\x0boutput_name\x18\x02 \x01(\tR\noutputName\x12\x30\n\tdata_type\x18\x03 \x01(\x0e\x32\x13.inference.DataTypeR\x08\x64\x61taType\x12\x12\n\x04\x64ims\x18\x04 \x03(\x03R\x04\x64ims\x12R\n\rinitial_state\x18\x05 \x03(\x0b\x32-.inference.ModelSequenceBatching.InitialStateR\x0cinitialState\x12\x45\n use_same_buffer_for_input_output\x18\x06 \x01(\x08R\x1buseSameBufferForInputOutput\x12.\n\x13use_growable_memory\x18\x07 \x01(\x08R\x11useGrowableMemory\x1a\x8b\x01\n\x0eStrategyDirect\x12?\n\x1cmax_queue_delay_microseconds\x18\x01 \x01(\x04R\x19maxQueueDelayMicroseconds\x12\x38\n\x18minimum_slot_utilization\x18\x02 \x01(\x02R\x16minimumSlotUtilization\x1a\xe8\x01\n\x0eStrategyOldest\x12\x36\n\x17max_candidate_sequences\x18\x01 \x01(\x05R\x15maxCandidateSequences\x12\x30\n\x14preferred_batch_size\x18\x02 \x03(\x05R\x12preferredBatchSize\x12?\n\x1cmax_queue_delay_microseconds\x18\x03 \x01(\x04R\x19maxQueueDelayMicroseconds\x12+\n\x11preserve_ordering\x18\x04 \x01(\x08R\x10preserveOrderingB\x11\n\x0fstrategy_choice\"\xd2\x03\n\x0fModelEnsembling\x12\x33\n\x04step\x18\x01 \x03(\x0b\x32\x1f.inference.ModelEnsembling.StepR\x04step\x1a\x89\x03\n\x04Step\x12\x1d\n\nmodel_name\x18\x01 \x01(\tR\tmodelName\x12#\n\rmodel_version\x18\x02 \x01(\x03R\x0cmodelVersion\x12J\n\tinput_map\x18\x03 \x03(\x0b\x32-.inference.ModelEnsembling.Step.InputMapEntryR\x08inputMap\x12M\n\noutput_map\x18\x04 \x03(\x0b\x32..inference.ModelEnsembling.Step.OutputMapEntryR\toutputMap\x12\'\n\x0fmodel_namespace\x18\x05 \x01(\tR\x0emodelNamespace\x1a;\n\rInputMapEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\x1a<\n\x0eOutputMapEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\"3\n\x0eModelParameter\x12!\n\x0cstring_value\x18\x01 \x01(\tR\x0bstringValue\"\xba\x03\n\x0bModelWarmup\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x1d\n\nbatch_size\x18\x02 \x01(\rR\tbatchSize\x12:\n\x06inputs\x18\x03 \x03(\x0b\x32\".inference.ModelWarmup.InputsEntryR\x06inputs\x12\x14\n\x05\x63ount\x18\x04 \x01(\rR\x05\x63ount\x1a\xcc\x01\n\x05Input\x12\x30\n\tdata_type\x18\x01 \x01(\x0e\x32\x13.inference.DataTypeR\x08\x64\x61taType\x12\x12\n\x04\x64ims\x18\x02 \x03(\x03R\x04\x64ims\x12\x1d\n\tzero_data\x18\x03 \x01(\x08H\x00R\x08zeroData\x12!\n\x0brandom_data\x18\x04 \x01(\x08H\x00R\nrandomData\x12(\n\x0finput_data_file\x18\x05 \x01(\tH\x00R\rinputDataFileB\x11\n\x0finput_data_type\x1aW\n\x0bInputsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x32\n\x05value\x18\x02 \x01(\x0b\x32\x1c.inference.ModelWarmup.InputR\x05value:\x02\x38\x01\"A\n\x0fModelOperations\x12.\n\x13op_library_filename\x18\x01 \x03(\tR\x11opLibraryFilename\"6\n\x16ModelTransactionPolicy\x12\x1c\n\tdecoupled\x18\x01 \x01(\x08R\tdecoupled\"\x8c\x02\n\x15ModelRepositoryAgents\x12>\n\x06\x61gents\x18\x01 \x03(\x0b\x32&.inference.ModelRepositoryAgents.AgentR\x06\x61gents\x1a\xb2\x01\n\x05\x41gent\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12V\n\nparameters\x18\x02 \x03(\x0b\x32\x36.inference.ModelRepositoryAgents.Agent.ParametersEntryR\nparameters\x1a=\n\x0fParametersEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\",\n\x12ModelResponseCache\x12\x16\n\x06\x65nable\x18\x01 \x01(\x08R\x06\x65nable\"\xbe\r\n\x0bModelConfig\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x1a\n\x08platform\x18\x02 \x01(\tR\x08platform\x12\x18\n\x07\x62\x61\x63kend\x18\x11 \x01(\tR\x07\x62\x61\x63kend\x12\x18\n\x07runtime\x18\x19 \x01(\tR\x07runtime\x12\x44\n\x0eversion_policy\x18\x03 \x01(\x0b\x32\x1d.inference.ModelVersionPolicyR\rversionPolicy\x12$\n\x0emax_batch_size\x18\x04 \x01(\x05R\x0cmaxBatchSize\x12+\n\x05input\x18\x05 \x03(\x0b\x32\x15.inference.ModelInputR\x05input\x12.\n\x06output\x18\x06 \x03(\x0b\x32\x16.inference.ModelOutputR\x06output\x12\x36\n\x0b\x62\x61tch_input\x18\x14 \x03(\x0b\x32\x15.inference.BatchInputR\nbatchInput\x12\x39\n\x0c\x62\x61tch_output\x18\x15 \x03(\x0b\x32\x16.inference.BatchOutputR\x0b\x62\x61tchOutput\x12\x46\n\x0coptimization\x18\x0c \x01(\x0b\x32\".inference.ModelOptimizationPolicyR\x0coptimization\x12L\n\x10\x64ynamic_batching\x18\x0b \x01(\x0b\x32\x1f.inference.ModelDynamicBatchingH\x00R\x0f\x64ynamicBatching\x12O\n\x11sequence_batching\x18\r \x01(\x0b\x32 .inference.ModelSequenceBatchingH\x00R\x10sequenceBatching\x12M\n\x13\x65nsemble_scheduling\x18\x0f \x01(\x0b\x32\x1a.inference.ModelEnsemblingH\x00R\x12\x65nsembleScheduling\x12\x44\n\x0einstance_group\x18\x07 \x03(\x0b\x32\x1d.inference.ModelInstanceGroupR\rinstanceGroup\x12\x34\n\x16\x64\x65\x66\x61ult_model_filename\x18\x08 \x01(\tR\x14\x64\x65\x66\x61ultModelFilename\x12Z\n\x12\x63\x63_model_filenames\x18\t \x03(\x0b\x32,.inference.ModelConfig.CcModelFilenamesEntryR\x10\x63\x63ModelFilenames\x12G\n\x0bmetric_tags\x18\n \x03(\x0b\x32&.inference.ModelConfig.MetricTagsEntryR\nmetricTags\x12\x46\n\nparameters\x18\x0e \x03(\x0b\x32&.inference.ModelConfig.ParametersEntryR\nparameters\x12\x39\n\x0cmodel_warmup\x18\x10 \x03(\x0b\x32\x16.inference.ModelWarmupR\x0bmodelWarmup\x12\x45\n\x10model_operations\x18\x12 \x01(\x0b\x32\x1a.inference.ModelOperationsR\x0fmodelOperations\x12[\n\x18model_transaction_policy\x18\x13 \x01(\x0b\x32!.inference.ModelTransactionPolicyR\x16modelTransactionPolicy\x12X\n\x17model_repository_agents\x18\x17 \x01(\x0b\x32 .inference.ModelRepositoryAgentsR\x15modelRepositoryAgents\x12\x44\n\x0eresponse_cache\x18\x18 \x01(\x0b\x32\x1d.inference.ModelResponseCacheR\rresponseCache\x1a\x43\n\x15\x43\x63ModelFilenamesEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\x1a=\n\x0fMetricTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\x1aX\n\x0fParametersEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12/\n\x05value\x18\x02 \x01(\x0b\x32\x19.inference.ModelParameterR\x05value:\x02\x38\x01\x42\x13\n\x11scheduling_choice*\xfa\x01\n\x08\x44\x61taType\x12\x10\n\x0cTYPE_INVALID\x10\x00\x12\r\n\tTYPE_BOOL\x10\x01\x12\x0e\n\nTYPE_UINT8\x10\x02\x12\x0f\n\x0bTYPE_UINT16\x10\x03\x12\x0f\n\x0bTYPE_UINT32\x10\x04\x12\x0f\n\x0bTYPE_UINT64\x10\x05\x12\r\n\tTYPE_INT8\x10\x06\x12\x0e\n\nTYPE_INT16\x10\x07\x12\x0e\n\nTYPE_INT32\x10\x08\x12\x0e\n\nTYPE_INT64\x10\t\x12\r\n\tTYPE_FP16\x10\n\x12\r\n\tTYPE_FP32\x10\x0b\x12\r\n\tTYPE_FP64\x10\x0c\x12\x0f\n\x0bTYPE_STRING\x10\r\x12\r\n\tTYPE_BF16\x10\x0e\x42t\n\rcom.inferenceB\x10ModelConfigProtoP\x01Z\r./grpc-client\xa2\x02\x03IXX\xaa\x02\tInference\xca\x02\tInference\xe2\x02\x15Inference\\GPBMetadata\xea\x02\tInferenceb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'model_config_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\rcom.inferenceB\020ModelConfigProtoP\001Z\r./grpc-client\242\002\003IXX\252\002\tInference\312\002\tInference\342\002\025Inference\\GPBMetadata\352\002\tInference'
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_LOWERBOUND_INPUTENTRY']._options = None
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_LOWERBOUND_INPUTENTRY']._serialized_options = b'8\001'
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_INPUTENTRY']._options = None
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_INPUTENTRY']._serialized_options = b'8\001'
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS_ACCELERATOR_PARAMETERSENTRY']._options = None
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS_ACCELERATOR_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_MODELDYNAMICBATCHING_PRIORITYQUEUEPOLICYENTRY']._options = None
  _globals['_MODELDYNAMICBATCHING_PRIORITYQUEUEPOLICYENTRY']._serialized_options = b'8\001'
  _globals['_MODELENSEMBLING_STEP_INPUTMAPENTRY']._options = None
  _globals['_MODELENSEMBLING_STEP_INPUTMAPENTRY']._serialized_options = b'8\001'
  _globals['_MODELENSEMBLING_STEP_OUTPUTMAPENTRY']._options = None
  _globals['_MODELENSEMBLING_STEP_OUTPUTMAPENTRY']._serialized_options = b'8\001'
  _globals['_MODELWARMUP_INPUTSENTRY']._options = None
  _globals['_MODELWARMUP_INPUTSENTRY']._serialized_options = b'8\001'
  _globals['_MODELREPOSITORYAGENTS_AGENT_PARAMETERSENTRY']._options = None
  _globals['_MODELREPOSITORYAGENTS_AGENT_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_MODELCONFIG_CCMODELFILENAMESENTRY']._options = None
  _globals['_MODELCONFIG_CCMODELFILENAMESENTRY']._serialized_options = b'8\001'
  _globals['_MODELCONFIG_METRICTAGSENTRY']._options = None
  _globals['_MODELCONFIG_METRICTAGSENTRY']._serialized_options = b'8\001'
  _globals['_MODELCONFIG_PARAMETERSENTRY']._options = None
  _globals['_MODELCONFIG_PARAMETERSENTRY']._serialized_options = b'8\001'
  _globals['_DATATYPE']._serialized_start=10415
  _globals['_DATATYPE']._serialized_end=10665
  _globals['_MODELRATELIMITER']._serialized_start=34
  _globals['_MODELRATELIMITER']._serialized_end=226
  _globals['_MODELRATELIMITER_RESOURCE']._serialized_start=150
  _globals['_MODELRATELIMITER_RESOURCE']._serialized_end=226
  _globals['_MODELINSTANCEGROUP']._serialized_start=229
  _globals['_MODELINSTANCEGROUP']._serialized_end=850
  _globals['_MODELINSTANCEGROUP_SECONDARYDEVICE']._serialized_start=611
  _globals['_MODELINSTANCEGROUP_SECONDARYDEVICE']._serialized_end=783
  _globals['_MODELINSTANCEGROUP_SECONDARYDEVICE_SECONDARYDEVICEKIND']._serialized_start=746
  _globals['_MODELINSTANCEGROUP_SECONDARYDEVICE_SECONDARYDEVICEKIND']._serialized_end=783
  _globals['_MODELINSTANCEGROUP_KIND']._serialized_start=785
  _globals['_MODELINSTANCEGROUP_KIND']._serialized_end=850
  _globals['_MODELTENSORRESHAPE']._serialized_start=852
  _globals['_MODELTENSORRESHAPE']._serialized_end=894
  _globals['_MODELINPUT']._serialized_start=897
  _globals['_MODELINPUT']._serialized_end=1285
  _globals['_MODELINPUT_FORMAT']._serialized_start=1226
  _globals['_MODELINPUT_FORMAT']._serialized_end=1285
  _globals['_MODELOUTPUT']._serialized_start=1288
  _globals['_MODELOUTPUT']._serialized_end=1527
  _globals['_BATCHINPUT']._serialized_start=1530
  _globals['_BATCHINPUT']._serialized_end=1916
  _globals['_BATCHINPUT_KIND']._serialized_start=1711
  _globals['_BATCHINPUT_KIND']._serialized_end=1916
  _globals['_BATCHOUTPUT']._serialized_start=1919
  _globals['_BATCHOUTPUT']._serialized_end=2093
  _globals['_BATCHOUTPUT_KIND']._serialized_start=2051
  _globals['_BATCHOUTPUT_KIND']._serialized_end=2093
  _globals['_MODELVERSIONPOLICY']._serialized_start=2096
  _globals['_MODELVERSIONPOLICY']._serialized_end=2414
  _globals['_MODELVERSIONPOLICY_LATEST']._serialized_start=2307
  _globals['_MODELVERSIONPOLICY_LATEST']._serialized_end=2350
  _globals['_MODELVERSIONPOLICY_ALL']._serialized_start=2352
  _globals['_MODELVERSIONPOLICY_ALL']._serialized_end=2357
  _globals['_MODELVERSIONPOLICY_SPECIFIC']._serialized_start=2359
  _globals['_MODELVERSIONPOLICY_SPECIFIC']._serialized_end=2397
  _globals['_MODELOPTIMIZATIONPOLICY']._serialized_start=2417
  _globals['_MODELOPTIMIZATIONPOLICY']._serialized_end=4567
  _globals['_MODELOPTIMIZATIONPOLICY_GRAPH']._serialized_start=3076
  _globals['_MODELOPTIMIZATIONPOLICY_GRAPH']._serialized_end=3105
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA']._serialized_start=3108
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA']._serialized_end=3941
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC']._serialized_start=3311
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC']._serialized_end=3941
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_SHAPE']._serialized_start=3545
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_SHAPE']._serialized_end=3570
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_LOWERBOUND']._serialized_start=3573
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_LOWERBOUND']._serialized_end=3826
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_LOWERBOUND_INPUTENTRY']._serialized_start=3713
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_LOWERBOUND_INPUTENTRY']._serialized_end=3826
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_INPUTENTRY']._serialized_start=3713
  _globals['_MODELOPTIMIZATIONPOLICY_CUDA_GRAPHSPEC_INPUTENTRY']._serialized_end=3826
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS']._serialized_start=3944
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS']._serialized_end=4446
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS_ACCELERATOR']._serialized_start=4232
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS_ACCELERATOR']._serialized_end=4446
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS_ACCELERATOR_PARAMETERSENTRY']._serialized_start=4385
  _globals['_MODELOPTIMIZATIONPOLICY_EXECUTIONACCELERATORS_ACCELERATOR_PARAMETERSENTRY']._serialized_end=4446
  _globals['_MODELOPTIMIZATIONPOLICY_PINNEDMEMORYBUFFER']._serialized_start=4448
  _globals['_MODELOPTIMIZATIONPOLICY_PINNEDMEMORYBUFFER']._serialized_end=4492
  _globals['_MODELOPTIMIZATIONPOLICY_MODELPRIORITY']._serialized_start=4494
  _globals['_MODELOPTIMIZATIONPOLICY_MODELPRIORITY']._serialized_end=4567
  _globals['_MODELQUEUEPOLICY']._serialized_start=4570
  _globals['_MODELQUEUEPOLICY']._serialized_end=4868
  _globals['_MODELQUEUEPOLICY_TIMEOUTACTION']._serialized_start=4830
  _globals['_MODELQUEUEPOLICY_TIMEOUTACTION']._serialized_end=4868
  _globals['_MODELDYNAMICBATCHING']._serialized_start=4871
  _globals['_MODELDYNAMICBATCHING']._serialized_end=5438
  _globals['_MODELDYNAMICBATCHING_PRIORITYQUEUEPOLICYENTRY']._serialized_start=5339
  _globals['_MODELDYNAMICBATCHING_PRIORITYQUEUEPOLICYENTRY']._serialized_end=5438
  _globals['_MODELSEQUENCEBATCHING']._serialized_start=5441
  _globals['_MODELSEQUENCEBATCHING']._serialized_end=7276
  _globals['_MODELSEQUENCEBATCHING_CONTROL']._serialized_start=5879
  _globals['_MODELSEQUENCEBATCHING_CONTROL']._serialized_end=6246
  _globals['_MODELSEQUENCEBATCHING_CONTROL_KIND']._serialized_start=6129
  _globals['_MODELSEQUENCEBATCHING_CONTROL_KIND']._serialized_end=6246
  _globals['_MODELSEQUENCEBATCHING_CONTROLINPUT']._serialized_start=6248
  _globals['_MODELSEQUENCEBATCHING_CONTROLINPUT']._serialized_end=6350
  _globals['_MODELSEQUENCEBATCHING_INITIALSTATE']._serialized_start=6353
  _globals['_MODELSEQUENCEBATCHING_INITIALSTATE']._serialized_end=6533
  _globals['_MODELSEQUENCEBATCHING_STATE']._serialized_start=6536
  _globals['_MODELSEQUENCEBATCHING_STATE']._serialized_end=6880
  _globals['_MODELSEQUENCEBATCHING_STRATEGYDIRECT']._serialized_start=6883
  _globals['_MODELSEQUENCEBATCHING_STRATEGYDIRECT']._serialized_end=7022
  _globals['_MODELSEQUENCEBATCHING_STRATEGYOLDEST']._serialized_start=7025
  _globals['_MODELSEQUENCEBATCHING_STRATEGYOLDEST']._serialized_end=7257
  _globals['_MODELENSEMBLING']._serialized_start=7279
  _globals['_MODELENSEMBLING']._serialized_end=7745
  _globals['_MODELENSEMBLING_STEP']._serialized_start=7352
  _globals['_MODELENSEMBLING_STEP']._serialized_end=7745
  _globals['_MODELENSEMBLING_STEP_INPUTMAPENTRY']._serialized_start=7624
  _globals['_MODELENSEMBLING_STEP_INPUTMAPENTRY']._serialized_end=7683
  _globals['_MODELENSEMBLING_STEP_OUTPUTMAPENTRY']._serialized_start=7685
  _globals['_MODELENSEMBLING_STEP_OUTPUTMAPENTRY']._serialized_end=7745
  _globals['_MODELPARAMETER']._serialized_start=7747
  _globals['_MODELPARAMETER']._serialized_end=7798
  _globals['_MODELWARMUP']._serialized_start=7801
  _globals['_MODELWARMUP']._serialized_end=8243
  _globals['_MODELWARMUP_INPUT']._serialized_start=7950
  _globals['_MODELWARMUP_INPUT']._serialized_end=8154
  _globals['_MODELWARMUP_INPUTSENTRY']._serialized_start=8156
  _globals['_MODELWARMUP_INPUTSENTRY']._serialized_end=8243
  _globals['_MODELOPERATIONS']._serialized_start=8245
  _globals['_MODELOPERATIONS']._serialized_end=8310
  _globals['_MODELTRANSACTIONPOLICY']._serialized_start=8312
  _globals['_MODELTRANSACTIONPOLICY']._serialized_end=8366
  _globals['_MODELREPOSITORYAGENTS']._serialized_start=8369
  _globals['_MODELREPOSITORYAGENTS']._serialized_end=8637
  _globals['_MODELREPOSITORYAGENTS_AGENT']._serialized_start=8459
  _globals['_MODELREPOSITORYAGENTS_AGENT']._serialized_end=8637
  _globals['_MODELREPOSITORYAGENTS_AGENT_PARAMETERSENTRY']._serialized_start=4385
  _globals['_MODELREPOSITORYAGENTS_AGENT_PARAMETERSENTRY']._serialized_end=4446
  _globals['_MODELRESPONSECACHE']._serialized_start=8639
  _globals['_MODELRESPONSECACHE']._serialized_end=8683
  _globals['_MODELCONFIG']._serialized_start=8686
  _globals['_MODELCONFIG']._serialized_end=10412
  _globals['_MODELCONFIG_CCMODELFILENAMESENTRY']._serialized_start=10171
  _globals['_MODELCONFIG_CCMODELFILENAMESENTRY']._serialized_end=10238
  _globals['_MODELCONFIG_METRICTAGSENTRY']._serialized_start=10240
  _globals['_MODELCONFIG_METRICTAGSENTRY']._serialized_end=10301
  _globals['_MODELCONFIG_PARAMETERSENTRY']._serialized_start=10303
  _globals['_MODELCONFIG_PARAMETERSENTRY']._serialized_end=10391
# @@protoc_insertion_point(module_scope)
