# """An example of workflow"""
# import random
# from typing import Union
#
# from llama_index.core.bridge.pydantic import BaseModel, Field
# from llama_index.core.workflow import (
#     Context,
#     Event,
#     HumanResponseEvent,
#     InputRequiredEvent,
#     StartEvent,
#     StopEvent,
#     Workflow,
#     step,
# )
# from llama_index.utils.workflow import draw_all_possible_flows
#
# from src.llama_index_experiments.llm_load import llm
# from src.utils.prompts import validation_prompt_template
#
#
# class QueryValidationFormat(BaseModel):
#     """Format used to get llm validation output"""
#
#     query: str = Field(description="The requested query")
#     judgment: bool = Field(description="The judgment")
#     reason: str = Field(description="The reason for the judgment")
#
#
# class RetryEvent(Event):
#     """aaa"""
#
#     query: str
#
#
# class FailedEvent(Event):
#     """aaa"""
#
#     error: str
#
#
# class AnswerQuery(Event):
#     """aaa"""
#
#     query: str
#
#
# class ProgressEvent(Event):
#     """aaa"""
#
#     msg: str
#
#
# class HfWorkflow(Workflow):
#     """aaa"""
#
#     @step(pass_context=True)
#     async def ask_query(self,
#                         ctx: Context,
#                         ev: Union[StartEvent, RetryEvent]) -> AnswerQuery:
#         """aaa"""
#         question = ev.query
#         human_query = await ctx.wait_for_event(
#             HumanResponseEvent,
#             waiter_id=question,
#             waiter_event=InputRequiredEvent(prefix=question, user_name="Human"),
#             requirements={"user_name": "Human"},
#          )
#
#         ctx.write_event_to_stream(
#             ProgressEvent(msg=f"The human as responded: {human_query}")
#         )
#
#         ctx.data["query"] = human_query
#         return AnswerQuery(query=question)
#
#     @step(pass_context=True)
#     async def get_query(
#                         self,
#                         ctx: Context,
#                         ev: AnswerQuery
#     ) -> Union[FailedEvent, StopEvent]:
#         """aaa"""
#
#         print(ctx.data["query"])
#
#         query_validation = llm\
#             .as_structured_llm(QueryValidationFormat)\
#             .complete(validation_prompt_template.format(query_str=ev.query))\
#             .raw
#
#         print(f'query: {query_validation.query}')
#         print(f'judgment: {query_validation.judgment}')
#         print(f'judgment: {type(query_validation.judgment)}')
#         print(f'reason: {query_validation.reason}')
#
#         if not query_validation.judgment:
#             return FailedEvent(error="Failed to answer the query")
#
#         return StopEvent(result="I can answer your query")
#
#     @step()
#     async def improve_query(self,
#                             ev: FailedEvent) -> Union[RetryEvent,
#                                                       StopEvent]:
#         """aaa"""
#         error = ev.error
#         print(error)
#         random_number = random.randint(0, 1)
#         if random_number == 0:
#             return RetryEvent(query="Here's a better query")
#
#         return StopEvent(result="Your query cannot be fixed")
#
#
# async def workflow_execution():
#     """Main function"""
#     w = HfWorkflow(timeout=10, verbose=False)
#     handler = await w.run(query="aaa")
#
#     async for event in handler.stream_events():
#         if isinstance(event, InputRequiredEvent):
#             # capture keyboard input
#             response = input(event.prefix)
#             # send our response back
#             handler.ctx.send_event(
#                 HumanResponseEvent(
#                     response=response
#                 )
#             )
#
#     response = await handler
#     print(str(response))
#
#
# if __name__ == "__main__":
#     draw_all_possible_flows(HfWorkflow(), filename="./docs/img/MyWorkFlow.html")
