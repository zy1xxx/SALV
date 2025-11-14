# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen

import json,copy
if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )

        special_mode=None
        if "<split_token>" in chosen_messages[1]['content'] and "<split_token>" in rejected_messages[1]['content']:
            special_mode="mask mode"
        elif "<select_token>" in chosen_messages[1]['content'] and "<select_token>" in rejected_messages[1]['content']:
            special_mode="select mode"
        
        if special_mode=="mask mode":
            #added by zy
            split_chosen=chosen_messages[1]['content'].split("<split_token>")
            if len(split_chosen)==2:
                chosen_messages[1]['content']=split_chosen[0]
                chosen_mask_index=json.loads(split_chosen[1])
            else:
                chosen_mask_index=[]
            split_reject=rejected_messages[1]['content'].split("<split_token>")
            if len(split_reject)==2:
                rejected_messages[1]['content']=split_reject[0]
                reject_mask_index=json.loads(split_reject[1])
            else:
                reject_mask_index=[]
        elif special_mode=="select mode":
            split_chosen=chosen_messages[1]['content'].split("<select_token>")
            if len(split_chosen)==2:
                chosen_messages[1]['content']=split_chosen[0]
                chosen_select_index=json.loads(split_chosen[1])
            else:
                chosen_select_index=[]
            split_reject=rejected_messages[1]['content'].split("<select_token>")
            if len(split_reject)==2:
                rejected_messages[1]['content']=split_reject[0]
                reject_select_index=json.loads(split_reject[1])
            else:
                reject_select_index=[]

        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len)
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]
        
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        #added by zy
        if special_mode=="mask mode":
            chosen_ids_mask=copy.deepcopy(chosen_ids)
            for s,e in chosen_mask_index:
                for index in range(s,e):
                    if index<len(chosen_ids):
                        chosen_ids_mask[index]=IGNORE_INDEX
            reject_ids_mask=copy.deepcopy(rejected_ids)
            for s,e in reject_mask_index:
                for index in range(s,e):
                    if index<len(rejected_ids):
                        reject_ids_mask[index]=IGNORE_INDEX
        
            chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids_mask
            rejected_labels = [IGNORE_INDEX] * source_len + reject_ids_mask
        elif special_mode=="select mode":
            chosen_ids_select=[IGNORE_INDEX] * len(chosen_ids)
            for s,e in chosen_select_index:
                for index in range(s,e):
                    if index<len(chosen_ids):
                        chosen_ids_select[index]=chosen_ids[index]
            reject_ids_select=[IGNORE_INDEX] * len(rejected_ids)
            for s,e in reject_select_index:
                for index in range(s,e):
                    if index<len(rejected_ids):
                        reject_ids_select[index]=rejected_ids[index]
            chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids_select
            rejected_labels = [IGNORE_INDEX] * source_len + reject_ids_select
        else:
            chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
            rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
