import json
import logging

import openai

from interface import prompt_text
from utils.utils import print_gpt_process
from ..entities.namespace import Namespace


class Session:
    namespace: Namespace = None

    messages: list[dict] = []

    model: str = "gpt-3.5-turbo-0613"

    def __init__(self, modules: list, model: str = "gpt-3.5-turbo-16k", **kwargs):
        self.namespace = Namespace(modules)
        self.model = model
        self.messages.append(
            {
                "role": "system",
                "content": prompt_text.system_prompt
            }
        )
        self.resp_log = []

        self.args = {
            "model": self.model,
            "messages": self.messages,
            **kwargs
        }
        if len(self.namespace.functions_list) > 0:
            self.args['functions'] = self.namespace.functions_list
            self.args['function_call'] = "auto"

    def ask(self, msg: str, fc_chain: bool = True) -> dict:
        self.messages.append(
            {
                "role": "user",
                "content": msg
            }
        )

        print_gpt_process(msg, 'msg_received')

        resp = openai.ChatCompletion.create(
            **self.args
        )
        self.resp_log.append(resp)

        logging.debug("Response: {}".format(resp))
        reply_msg = resp["choices"][0]['message']

        ret = {}

        if fc_chain:
            while 'function_call' in reply_msg:
                resp = self.fc_chain(reply_msg['function_call'])
                reply_msg = resp["choices"][0]['message']
            ret = {
                "type": "message",
                "value": reply_msg['content'],
            }

            self.messages.append({
                "role": "assistant",
                "content": reply_msg['content']
            })

            print_gpt_process(reply_msg['content'], 'msg_replied')

            return ret['value']

        else:
            if 'function_call' in reply_msg:

                fc = reply_msg['function_call']
                args = json.loads(fc['arguments'])
                call_ret = self._call_function(fc['name'], args)

                self.messages.append({
                    "role": "function",
                    "name": fc['name'],
                    "content": str(call_ret)
                })

                ret = {
                    "type": "function_call",
                    "func": fc['name'].replace('-', '.'),
                    "value": call_ret,
                }
            else:
                ret = {
                    "type": "message",
                    "value": reply_msg['content'],
                }

                self.messages.append({
                    "role": "assistant",
                    "content": reply_msg['content']
                })

            return ret['value']

    def fc_chain(self, fc_cmd: dict):
        """
        Excecute the function call and return the result to ChatGPT.

        Args:
            fc_cmd(dict): The function call command.

        Returns:
            dict: The response from ChatGPT.
        """
        fc_args = json.loads(fc_cmd['arguments'])
        content = f"function name:\n" \
                  f"{fc_cmd['name']}\n\n" \
                  f"args:\n" \
                  f"{fc_cmd['arguments']}"
        print_gpt_process(content, 'func_called')

        call_ret = self._call_function(fc_cmd['name'], fc_args)
        print_gpt_process(f"ret_value:\n{call_ret}", 'func_completed')

        self.messages.append({
            "role": "function",
            "name": fc_cmd['name'],
            "content": str(call_ret)
        })
        resp = openai.ChatCompletion.create(
            **self.args
        )
        self.resp_log.append(resp)

        return resp

    def _call_function(self, function_name: str, args: dict):
        return self.namespace.call_function(function_name, args)
