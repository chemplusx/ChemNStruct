import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Union

from src.schemas.ChemStruct import ChemStruct
from src.schemas.misc import Record
from src.utils.train_utils import create_output_from_entities, MODEL_INPUT_TEMPLATE
from src.utils.utils import load_lines, parse_jsonl

ENTITY_TYPES = ['chemical']
ENTITY_DEFENITIONS = [
    'mention of drugs, organic compounds, protiens and any other chemical substances'
]
INSTRUCTION_TEXT = "You are solving the NER problem. Extract from the text words related to each of the following entities: chemical"


class AnnotatedDataset(Record):
    __attributes__ = ['file_name', 'text', 'sentence_id', 'entities']

    def __init__(self, file_name, text, sentence_id, entities):
        self.file_name = file_name
        self.text = text
        self.sentence_id = sentence_id
        self.entities = entities


class RuDReCEntity(Record):
    __attributes__ = [
        'entity_id', 'entity_text', 'entity_type',
        'start', 'end', 'concept_id', 'concept_name'
    ]

    def __init__(self, entity_id, entity_text, entity_type, start, end, concept_id, concept_name):
        self.entity_id = entity_id
        self.entity_text = entity_text
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.concept_id = concept_id
        self.concept_name = concept_name


def parse_entities(items):
    for item in items:
        yield RuDReCEntity(
            item['entity_id'],
            item['entity_text'],
            item['entity_type'],
            item['start'],
            item['end'],
            item.get('concept_id'),
            item.get('concept_name')
        )


def parse_annotated_data(items):
    for item in items:
        entities = list(parse_entities(item['entities']))
        yield AnnotatedDataset(
            item['file_name'],
            item['text'],
            item['sentence_id'],
            entities
        )


def load_annotated_data(path):
    lines = load_lines(path)
    items = parse_jsonl(lines)
    return parse_annotated_data(items)



def entity_type_to_instruction(entity_type: str) -> str:
    base_phrase = 'You are solving the NER problem. Extract from the text '
    return base_phrase + dict(zip(ENTITY_TYPES, ENTITY_DEFENITIONS))[entity_type]


def parse_entities_from_record(record: AnnotatedDataset) -> tuple[str, dict[str, list]]:
    entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))
    for entity in record.entities:
        entities[entity.entity_type].append(entity.entity_text)

    return record.text, entities


def create_instructions_for_record(
        record: AnnotatedDataset,
        is_separate_labels: bool = False
) -> Union[list[ChemStruct], ChemStruct]:
    text, entities = parse_entities_from_record(record)
    if is_separate_labels:
        record_instructions = []
        for entity_type in entities.keys():
            instruction = entity_type_to_instruction(entity_type)
            record_instructions.append({
                'instruction': instruction,
                'input': text,
                'output': create_output_from_entities(entities[entity_type]),
                'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=instruction.strip(),
                                                                       inp=text.strip()),
                'label': entity_type,
                'id': f"{record.sentence_id}_{record.file_name}"
            })
        return record_instructions
    else:
        return {
            'instruction': INSTRUCTION_TEXT,
            'input': text,
            'output': create_output_from_entities(entities, out_type=2),
            'source': MODEL_INPUT_TEMPLATE['prompts_input'].format(instruction=INSTRUCTION_TEXT.strip(),
                                                                   inp=text.strip()),
            'raw_entities': entities,
            'id': f"{record.sentence_id}_{record.file_name}"
        }


def _fill_instructions_list(dataset: list[AnnotatedDataset], is_separate_labels: bool = False) -> list[ChemStruct]:
    instructions = []
    for record in tqdm(dataset):
        if is_separate_labels:
            instructions = np.concatenate((instructions, create_instructions_for_record(record, is_separate_labels)))
        else:
            instructions.append(create_instructions_for_record(record, is_separate_labels))

    return instructions


def create_instruct_dataset(data_path: str, max_instances: int = -1, is_separate_labels: bool = False) -> list[
    ChemStruct]:
    rudrec_dataset = list(load_annotated_data(data_path))

    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    return _fill_instructions_list(rudrec_dataset, is_separate_labels)


def create_train_test_instruct_datasets(
        data_path: str,
        max_instances: int = -1,
        is_separate_labels: bool = False,
        test_size: float = 0.3,
        random_seed: int = 42
) -> tuple[list[ChemStruct], list[ChemStruct]]:
    rudrec_dataset = list(load_annotated_data(data_path))

    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    train_dataset, test_dataset = train_test_split(rudrec_dataset, test_size=test_size, random_state=random_seed)
    return _fill_instructions_list(train_dataset, is_separate_labels), _fill_instructions_list(test_dataset,
                                                                                               is_separate_labels)
