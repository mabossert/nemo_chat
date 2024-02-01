from nemollm.api import NemoLLM
from jl_logging import LoggingMixin
import json
from gpudb import GPUdb


# from rich.logging import RichHandler
# from rich.console import Console
# from rich.theme import Theme


class NemoChatLLM(LoggingMixin):
    _api_host = "https://api.llm.ngc.nvidia.com/v1"
    _api_key = "NTdvMmcwdHRxdWNqNW05MTMyZzZidm1vNDoxOTRlY2E3Mi1lNmZhLTQ1MmMtOTY5OC0xZjZiNzY4Zjk3Y2M"
    _model_id = "gpt-43b-905"
    # mixtral 7b
    # _model_id="llama-2-70b-chat-hf"
    _max_tokens = 4096

    def __init__(self):
        self._conn = NemoLLM(
            api_host=self._api_host,
            api_key=self._api_key,
            # org_id=org_ID
        )

    def chat(self, in_ctx: list, question: str) -> list:
        new_ctx = in_ctx.copy()
        new_ctx.append(dict(role="user", content=question))
        self._print_last(new_ctx)

        response = self._conn.generate_chat(
            model=self._model_id,
            chat_context=new_ctx,
            logprobs=False,
            temperature=0,
            random_seed=0,
            repetition_penalty=1.
        )
        out_ctx = response['chat_context']
        self._print_last(out_ctx)

        last_prompt = out_ctx[-1]
        role = last_prompt['role']
        if (role != 'assistant'):
            raise ValueError("Got unexpected role: {role}")

        return out_ctx

    def _print_last(self, ctx: list) -> None:
        last_prompt = ctx[-1]
        role = last_prompt['role']
        content = last_prompt['content']

        response = self._conn.count_tokens_chat(model=self._model_id, chat_context=ctx)
        output_tokens = response['input_length']
        remaining_tokens = self._max_tokens - output_tokens

        self.log.info(f"{role}: {content.strip()} (tokens: {output_tokens}/{remaining_tokens})")


class SqlAssistLLM(LoggingMixin):
    # URL = 'http://g-p100-300-301-u29.tysons.kinetica.com:9191'
    # LOGIN = 'admin'
    # PASSWORD = 'Kinetica1!'

    URL = "https://demo72.kinetica.com/_gpudb"
    LOGIN = "demo"
    PASSWORD = "Kinetica123!"
    SQL_CONTEXT = "telecom.telecom_new_ctxt"

    def __init__(self):
        self._dbc = self._create_kdbc()

    def query(self, question: str) -> str:
        self.log.info(f"Query: {question}")
        sql = self._generate_sql(question)
        self.log.info(f"SQL: {sql}")
        response = self._execute_sql(sql)
        return json.dumps(response)

    def _generate_sql(self, question: str) -> str:
        sql_gen = f"GENERATE SQL FOR '{question}' WITH OPTIONS (context_name = '{self.SQL_CONTEXT}');"
        records = self._execute_sql(sql_gen)
        sql_response = records[0]['Response']
        return sql_response

    def _execute_sql(self, sql: str) -> list:
        response = self._dbc.execute_sql_and_decode(sql, limit=4, get_column_major=False)

        status_info = response['status_info']
        if (status_info['status'] != 'OK'):
            message = status_info['message']
            raise ValueError(message)

        records = response['records']
        if (len(records) == 0):
            raise ValueError("No records returned.")

        response_array = []
        for record in records:
            response_dict = {}
            for col, val in record.items():
                response_dict[col] = val
            response_array.append(response_dict)

        return response_array

    @classmethod
    def _create_kdbc(cls) -> GPUdb:
        options = GPUdb.Options()
        options.username = cls.LOGIN
        options.password = cls.PASSWORD
        options.skip_ssl_cert_verification = True
        options.disable_failover = True
        options.logging_level = 'INFO'
        kdbc = GPUdb(host=cls.URL, options=options)
        cls.logger().info(f"Connected to {kdbc.get_url()}. (version {str(kdbc.server_version)})")
        return kdbc


class KineticaLLM(LoggingMixin):
    #SA_PREFIX = 'SqlAssist:'
    SA_PREFIX = 'KineticaLLM | '

    def __init__(self):
        self._nemo = NemoChatLLM()
        self._sqlAssist = SqlAssistLLM()

    def chat(self, in_ctx: list, question: str) -> list:
        nemo_ctx = self._nemo.chat(in_ctx, question)
        last_prompt = nemo_ctx[-1]
        content = last_prompt['content'].strip()

        if (content.startswith(self.SA_PREFIX)):
            sa_question = content.removeprefix(self.SA_PREFIX).strip()
            sa_response = self._sqlAssist.query(sa_question)
            nemo_ctx = self._nemo.chat(in_ctx, f'{self.SA_PREFIX} {sa_response}')

        return nemo_ctx

