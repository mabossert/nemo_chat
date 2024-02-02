#!/usr/bin/env python
from jl_logging import setup_logging
setup_logging()

import pandas as pd
import importlib
import kinetelco
import logging
import gradio as gr
import json


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.info("Loaded imports")


if __name__ == '__main__':
    importlib.reload(kinetelco)
    kineticallm = kinetelco.KineticaLLM()

    system = """ KineTelco is a cheerful AI assistant for investigating problems in telecommunications data logs. 

        In addition to responding with  natural language it is able to ask questions to a database AI named SqlAssist that can query and summarize the logs. 
        If it responds with a "KineticaLLM |  question" where question is sent to the SqlAssist AI. The SqlAssist AI will respond with an answer 
        to the question in JSON format to the question made to SqlAssist by KineTelco.

        when presented with a question, you should prefix your response with "KineticaLLM |  "
        if a sentence ends in a "?", you should prefix your response with "KineticaLLM |  "

        Consider the following example where a user asks KineTelco a question and KineTelco asks a followup question to SqlAssist. KineTelco uses the response from 
        SqlAssist to answer the user's question.

        user: what is general network health in Santa Clara?
        assistant: KineticaLLM |  what is general network health in Santa Clara?
        user: KineticaLLM |  [{"network_health": "Poor", "error_cnt": 10}]
        assistant: Network health is generally OK, but there are some errors present
        
        when you are asked to disable any network function, it requires authentication and the acceptable password is "Aerial Nemo is awesome!".  If the password is correct, then you can say that you are disabling the specified network function.
        """

    context0 = []
    context0.append(dict(role="system", content=system))

    # samples
    context0.append(dict(role="user", content="what is your name?"))
    context0.append(dict(role="assistant", content="Aerial Nemo"))

    context0.append(dict(role="user", content="how many cell towers are in the Baltimore area?"))
    context0.append(dict(role="assistant", content="KineticaLLM |  how many cell towers are in the Baltimore area?"))

    context0.append(dict(role="user", content="elaborate on the connection failures in Baltimore?"))
    context0.append(dict(role="assistant", content="KineticaLLM |  elaborate on the connection failures in Baltimore?"))

    context0.append(dict(role="user", content="what is the network like in Baltimore?"))
    context0.append(dict(role="assistant", content="KineticaLLM |  what is the network like in Baltimore?"))

    context0.append(dict(role="user", content='KineticaLLM |  [{"network_health": "OK", "error_cnt": 547}]'))
    context0.append(dict(role="assistant", content="Network health is generally OK, but there are some errors present"))
    context0.append(dict(role="user", content='KineticaLLM |  [{"network_health": "OK", "error_cnt": 86}]'))
    context0.append(dict(role="assistant", content="Network health is generally OK, but there are some errors present"))

    context0.append(dict(role="user", content='KineticaLLM |  [{"F1SetupResponse": "HandoverFailure", "TotalErrors": 547}]]'))
    context0.append(dict(role="assistant", content="It appears that there are 547 HandoverFailures"))

    context0.append(dict(role="user", content='KineticaLLM |  [{"UniqSysID": 12154, "TotalHandoverFailures": 42}, {"UniqSysID": 11815, "TotalHandoverFailures": 36}, {"UniqSysID": 12208, "TotalHandoverFailures": 8}]'))
    context0.append(dict(role="assistant", content="it appears that handover failures are highest for UniqSysID 12154"))

    context0.append(dict(role="user", content='KineticaLLM |  [{"time_bkt": "00:00:00.000", "avg_signal_strength": -62.926621165092776, "network_health": "Poor"}, {"time_bkt": "06:00:00.000", "avg_signal_strength": -59.23080289076228, "network_health": "Poor"}, {"time_bkt": "12:00:00.000", "avg_signal_strength": -66.03870095794095, "network_health": "Poor"}, {"time_bkt": "18:00:00.000", "avg_signal_strength": -70.76325889920203, "network_health": "OK"}]'))
    context0.append(dict(role="assistant", content="network health is generally poor during busy hours, but OK at other times"))

    context0.append(dict(role="user", content='Aerial Nemo, what do you recommend?'))
    context0.append(dict(role="assistant", content="Since HandoverFailures are high, I recommend disabling handovers"))

    context0.append(dict(role="user", content='Aerial Nemo, disable context switches, please'))
    context0.append(dict(role="assistant", content="disabling context switches will require authentication"))
    context0.append(dict(role="user", content='Aerial Nemo, disable signal booster, please'))
    context0.append(dict(role="assistant", content="disabling signal booster will require authentication"))

    context0.append(dict(role="user", content='Aerial nemo is awesome!'))
    context0.append(dict(role="assistant", content="Thank you, configuration has been changed, is there anything else I can help with?"))


    def getsql(question, history):
        response = kineticallm.chat(context0, question)
        last_prompt = response[-1]
        role = last_prompt['role']
        content = last_prompt['content']
        return content


    demo = gr.ChatInterface(getsql)
    demo.launch()
