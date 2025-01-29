from uuid import uuid4

from models_client_python.chunker import Chunker, ChunkRequest
from models_client_python.common.host import Host, HostScheme
from models_client_python.common.requester import RequesterConfig

if __name__ == "__main__":
    MODEL_NAME = "chunker"
    MODEL_VERSION = "120252801110000"
    TEXTS = [
        "Hello, how are you?",
        """The Development and Reliability of 4 Clinical Neurocognitive Single-Leg Hop Tests: Implications for Return to Activity Decision-Making

Nathan Millikan
Dustin R. Grooms
Brett Hoffman
Janet E. Simon

###### Abstract
Context: Functional tests are limited primarily by measuring only physical performance. However, athletes often multitask, and deal with complex visual-spatial processing while being engaged in physical activity. _Objective_: To present the development and reliability of 4 new neurocognitive single-leg hop tests that provide more ecological validity to test sport activity demands than previous functional return to sport testing. _Design_: Cross-sectional. _Setting:_ Gymnasium. _Participants:_ Twenty-two healthy participants (9 males and 13 females; 20.9 [2.5] y, 171.2 [11.7] cm, 70.3 [11.0] kg) were recruited. _Interventions:_ Maximum distance (physical performance) and reaction time (cognitive performance) were measured for 3 of the neurocognitive hop tests all testing a different aspect of neurorecognition (single-leg _central-reaction_ hop--reaction time to 1 central stimulus, single-leg _peripheral-reaction_ crossover hop--reaction time between 2 peripheral stimuli, and single-leg _memory_ triple hop--reaction to memorized stimulus with distractor stimuli). Fastest time (physical performance) and reaction time (cognitive performance) were measured for the fourth neurocognitive hop test (single-leg _pursuit_ 6m hop--requiring visual field tracking [pursuit] and spatial navigation). _Main Outcome Measures:_ Intraclass correlation coefficients were calculated to assess reliability of the 4 new hop tests. Additionally, Bland-Altman plots and 1-sample $t$ tests were conducted for each single-leg neurocognitive hop to evaluate any systematic changes. _Results:_ Intraclass correlation coefficients based on day 1 and day 2 scores ranged from .87 to .98 for both legs for physical and cognitive performance. The Bland-Altman plots and 1-sample $t$ tests ($P\!>\!0.5$) indicated that all 4 single-leg neurocognitive hop tests did not change systematically. _Conclusions:_ These data provide evidence that a neurocognitive component can be added to the traditional single-leg hop tests to provide a more ecologically valid test that incorporates the integration of physical and cognitive function for return to sport. The test-retest reliability of the 4 new neurocognitive hop tests is highly reliable and does not change systematically.

functional performance, cognitive, visuomotor, FitLight 10.1123/jsr.2018-0037
""",
    ]

    ## Define gRPC requester config
    requester_config = RequesterConfig(host=Host(url="127.0.0.1", port="8001", scheme=HostScheme.http))

    ## Initialize Chunker
    chunker = Chunker.from_grpc(config=requester_config)

    ## Send request
    request = ChunkRequest(id=str(uuid4()), texts=TEXTS)
    response = chunker.chunk(model_name=MODEL_NAME, model_version=MODEL_VERSION, req=request)

    print(response)
