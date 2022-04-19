# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
import json
import numpy as np

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)


trex_relations = ['P178', 'P17', 'P569', 'P27', 'P36', 'P150', 'P1376', 'P47', 'P501', 'P131', 'P159', 'P103', 'P127',
                  'P138', 'P118', 'P831', 'P664', 'P54', 'P115', 'P279', 'P681', 'P527', 'P106', 'P570', 'P57', 'P58',
                  'P162', 'P344', 'P86', 'P1040', 'P161', 'P40', 'P22', 'P3373', 'P26', 'P25', 'P37', 'P1412', 'P364',
                  'P495', 'P449', 'P105', 'P20', 'P451', 'P735', 'P101', 'P136', 'P1441', 'P112', 'P706', 'P175',
                  'P641', 'P463', 'P1001', 'P155', 'P800', 'P171', 'P361', 'P39', 'P530', 'P19', 'P69', 'P194', 'P137',
                  'P607', 'P1313', 'P1906', 'P205', 'P264', 'P466', 'P170', 'P1431', 'P580', 'P582', 'P740', 'P50',
                  'P2578', 'P2579', 'P156', 'P30', 'P176', 'P166', 'P241', 'P413', 'P135', 'P195', 'P276', 'P180',
                  'P921', 'P1408', 'P1066', 'P802', 'P1038', 'P53', 'P102', 'P937', 'P2341', 'P828', 'P1542', 'P488',
                  'P807', 'P190', 'P611', 'P710', 'P2094', 'P765', 'P1454', 'P35', 'P577', 'P108', 'P172', 'P1303',
                  'P282', 'P885', 'P403', 'P1346', 'P6', 'P1344', 'P366', 'P1416', 'P749', 'P737', 'P559', 'P598',
                  'P609', 'P1923', 'P1071', 'P179', 'P870', 'P140', 'P647', 'P1336', 'P1435', 'P1876', 'P619', 'P123',
                  'P87', 'P747', 'P425', 'P551', 'P1532', 'P122', 'P461', 'P206', 'P585', 'P571', 'P931', 'P452',
                  'P355', 'P306', 'P915', 'P1589', 'P460', 'P1387', 'P81', 'P400', 'P1142', 'P1027', 'P2541', 'P682',
                  'P209', 'P674', 'P676', 'P199', 'P1889', 'P275', 'P974', 'P407', 'P1366', 'P412', 'P658', 'P371',
                  'P92', 'P1049', 'P750', 'P38', 'P277', 'P410', 'P1268', 'P469', 'P200', 'P201', 'P1411', 'P272',
                  'P157', 'P2439', 'P61', 'P1995', 'P509', 'P287', 'P16', 'P121', 'P1002', 'P2388', 'P2389', 'P2632',
                  'P1343', 'P2936', 'P2789', 'P427', 'P119', 'P576', 'P2176', 'P1196', 'P263', 'P1365', 'P457', 'P1830',
                  'P186', 'P832', 'P141', 'P2175', 'P840', 'P1877', 'P1056', 'P3450', 'P1269', 'P113', 'P533', 'P3448',
                  'P1191', 'P927', 'P610', 'P1327', 'P177', 'P1891', 'P169', 'P2670', 'P793', 'P770', 'P3137', 'P1383',
                  'P1064', 'P134', 'P945', 'P84', 'P1654', 'P2522', 'P1552', 'P1037', 'P286', 'P144', 'P689', 'P541',
                  'P991', 'P726', 'P780', 'P397', 'P398', 'P149', 'P1478', 'P98', 'P500', 'P1875', 'P2554', 'P59',
                  'P3461', 'P414', 'P748', 'P291', 'P85', 'P2348', 'P3320', 'P462', 'P1462', 'P2597', 'P2512', 'P1018',
                  'P21', 'P208', 'P2079', 'P1557', 'P1434', 'P1080', 'P1445', 'P1050', 'P3701', 'P767', 'P1299', 'P126',
                  'P360', 'P1304', 'P1029', 'P1672', 'P1582', 'P184', 'P2416', 'P65', 'P575', 'P3342', 'P3018', 'P183',
                  'P2546', 'P2499', 'P2500', 'P408', 'P450', 'P97', 'P417', 'P512', 'P1399', 'P404', 'P822', 'P941',
                  'P189', 'P725', 'P1619', 'P129', 'P629', 'P88', 'P2545', 'P1068', 'P1308', 'P1192', 'P2505', 'P376',
                  'P1535', 'P708', 'P1479', 'P2283', 'P1962', 'P2184', 'P163', 'P1419', 'P2286', 'P3190', 'P790',
                  'P1547', 'P1444', 'P504', 'P2596', 'P3095', 'P3300', 'P881', 'P1880', 'P358', 'P1427', 'P2438',
                  'P523', 'P524', 'P826', 'P485', 'P3679', 'P437', 'P553', 'P66', 'P2650', 'P816', 'P517', 'P1072',
                  'P78', 'P415', 'P825', 'P1302', 'P1716', 'P411', 'P734', 'P110', 'P1264', 'P289', 'P421', 'P2238',
                  'P375', 'P2989', 'P669', 'P2289', 'P111', 'P197', 'P620', 'P467', 'P3712', 'P185', 'P841', 'P739',
                  'P3301', 'P568', 'P567', 'P479', 'P625', 'P1433', 'P1429', 'P880', 'P1414', 'P547', 'P1731', 'P618',
                  'P2978', 'P1885', 'P516', 'P556', 'P522', 'P237', 'P1809', 'P2098', 'P1322', 'P3764', 'P2633',
                  'P1312', 'P859', 'P114', 'P2962', 'P1073', 'P1000', 'P1158', 'P196', 'P520', 'P2155', 'P606', 'P3403',
                  'P720']


class TREXProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return trex_relations

    def _create_examples(self, lines, dataset_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            # text_a: tokenized words
            text_a = line['token']
            # text_b: other information
            text_b = (line['subj_start'], line['subj_end'], line['obj_start'], line['obj_end'])
            label = line['relation']
            if label == 'no_relation' and dataset_type == 'train':
                continue
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


trex_relations_et = ['Q6465', 'Q4022', 'Q10742', 'Q15416', 'Q920182', 'Q1115575', 'Q169358', 'Q1344', 'Q1616075',
                     'Q8076', 'Q70208', 'Q11315', 'Q55488', 'Q482994', 'Q46831', 'Q7889', 'Q209939', 'Q622521',
                     'Q11303', 'Q1867183', 'Q532', 'Q18127', 'Q174989', 'Q131681', 'Q1077817', 'Q7278', 'Q2658935',
                     'Q618779', 'Q192299', 'Q1369421', 'Q644371', 'Q180673', 'Q159821', 'Q1750919', 'Q428661',
                     'Q484170', 'Q8054', 'Q23058', 'Q112099', 'Q341', 'Q838795', 'Q19020', 'Q1193246', 'Q7187',
                     'Q41710', 'Q33837', 'Q8047', 'Q212198', 'Q36649', 'Q60874', 'Q42998', 'Q694589', 'Q130695',
                     'Q732997', 'Q178790', 'Q816321', 'Q36534', 'Q3389302', 'Q55833', 'Q1153376', 'Q484416', 'Q272683',
                     'Q210167', 'Q131186', 'Q958314', 'Q713623', 'Q106658', 'Q613142', 'Q1620945', 'Q11288', 'Q2074737',
                     'Q37049', 'Q207694', 'Q1244922', 'Q9135', 'Q39367', 'Q1197685', 'Q276', 'Q11417', 'Q747074',
                     'Q167346', 'Q160738', 'Q32880', 'Q169930', 'Q35666', 'Q839954', 'Q16521', 'Q12280', 'Q2743',
                     'Q427626', 'Q1187580', 'Q39614', 'Q8274', 'Q5292', 'Q27686', 'Q170382', 'Q179872', 'Q159334',
                     'Q1011547', 'Q1248784', 'Q33384', 'Q101609', 'Q7075', 'Q9143', 'Q192283', 'Q1259759', 'Q1107679',
                     'Q12323', 'Q130003', 'Q581714', 'Q1137109', 'Q6368', 'Q1080794', 'Q188509', 'Q3220391', 'Q23413',
                     'Q11424', 'Q11173', 'Q188055', 'Q8513', 'Q210826', 'Q8928', 'Q202825', 'Q327333', 'Q222910',
                     'Q39715', 'Q134219', 'Q262166', 'Q220505', 'Q8072', 'Q1530705', 'Q26540', 'Q12140', 'Q142714',
                     'Q1768043', 'Q19317', 'Q196600', 'Q4671329', 'Q7969215', 'Q8192', 'Q105447', 'Q12136', 'Q8502',
                     'Q125928', 'Q628179', 'Q578521', 'Q737498', 'Q1070990', 'Q194195', 'Q378427', 'Q1137809', 'Q37756',
                     'Q4611891', 'Q23691', 'Q641066', 'Q108689', 'Q216337', 'Q223703', 'Q8366', 'Q41254', 'Q32815',
                     'Q163740', 'Q102356', 'Q1542651', 'Q968159', 'Q16338', 'Q45776', 'Q28803', 'Q3932296', 'Q24764',
                     'Q672729', 'Q131569', 'Q4421', 'Q154773', 'Q1066670', 'Q8514', 'Q336', 'Q179294', 'Q169872',
                     'Q44613', 'Q11353', 'Q699', 'Q215980', 'Q373899', 'Q559856', 'Q182547', 'Q872181', 'Q133056',
                     'Q2001305', 'Q3184121', 'Q35509', 'Q879050', 'Q229390', 'Q34274', 'Q226730', 'Q860582', 'Q506240',
                     'Q4508', 'Q7397', 'Q1107', 'Q5119', 'Q131669', 'Q188913', 'Q156362', 'Q223832', 'Q5356189',
                     'Q46721', 'Q3863', 'Q52105', 'Q202216', 'Q17451', 'Q79913', 'Q847017', 'Q11422', 'Q46395',
                     'Q4484477', 'Q637846', 'Q123480', 'Q879146', 'Q467745', 'Q24862', 'Q1134686', 'Q483463', 'Q192611',
                     'Q1044427', 'Q1615742', 'Q207320', 'Q179600', 'Q49084', 'Q137535', 'Q24746', 'Q225147', 'Q27448',
                     'Q265538', 'Q920890', 'Q134808', 'Q133215', 'Q7946', 'Q180874', 'Q179461', 'Q82673', 'Q133442',
                     'Q466704', 'Q54277', 'Q131734', 'Q44782', 'Q511056', 'Q1840161', 'Q87167', 'Q744099', 'Q188914',
                     'Q184188', 'Q174782', 'Q1952852', 'Q41425', 'Q9620', 'Q936518', 'Q748149', 'Q166247', 'Q5503',
                     'Q177456', 'Q475061', 'Q725169', 'Q422248', 'Q465299', 'Q9826', 'Q188451', 'Q612741', 'Q75520',
                     'Q37726', 'Q2389789', 'Q13741', 'Q7365', 'Q101998', 'Q6498903', 'Q1074076', 'Q11292', 'Q33215',
                     'Q41162', 'Q799469', 'Q61883', 'Q1234255', 'Q213907', 'Q622425', 'Q483110', 'Q211690', 'Q131257',
                     'Q22746', 'Q3745054', 'Q392316', 'Q1057954', 'Q494230', 'Q81163', 'Q22645', 'Q1857731', 'Q137341',
                     'Q2922711', 'Q204910', 'Q328468', 'Q2811', 'Q845945', 'Q672593', 'Q735', 'Q1343246', 'Q269770',
                     'Q375336', 'Q1254874', 'Q202686', 'Q582706', 'Q941818', 'Q7755', 'Q179435', 'Q132364', 'Q131436',
                     'Q937876', 'Q422106', 'Q1260006', 'Q52371', 'Q641226', 'Q245065', 'Q4418079', 'Q34038', 'Q12570',
                     'Q25295', 'Q260858', 'Q65943', 'Q40237', 'Q8795', 'Q9655', 'Q431603', 'Q507057', 'Q1429218',
                     'Q1760610', 'Q392928', 'Q140247', 'Q215655', 'Q184759', 'Q659103', 'Q39804', 'Q180126', 'Q23397',
                     'Q928830', 'Q12143', 'Q247073', 'Q1639634', 'Q182985', 'Q526877', 'Q428303', 'Q158218', 'Q33289',
                     'Q751705', 'Q282472', 'Q25368', 'Q234497', 'Q190928', 'Q49376', 'Q149918', 'Q190570', 'Q1643870',
                     'Q174736', 'Q736917', 'Q282', 'Q848797', 'Q5398426', 'Q233324', 'Q15284', 'Q1516659', 'Q171201',
                     'Q295469', 'Q235557', 'Q483373', 'Q204577', 'Q62832', 'Q310890', 'Q184356', 'Q220898', 'Q967098',
                     'Q181322', 'Q11344', 'Q82414', 'Q2989398', 'Q44377', 'Q187432', 'Q313301', 'Q875538', 'Q645883',
                     'Q921469', 'Q32096', 'Q5633421', 'Q499247', 'Q12876', 'Q188784', 'Q2706302', 'Q265868', 'Q41156',
                     'Q233591', 'Q25448', 'Q50399', 'Q180454', 'Q828160', 'Q9779', 'Q1172903', 'Q1268865', 'Q752783',
                     'Q17205', 'Q1002812', 'Q15089', 'Q428602', 'Q43742', 'Q11387', 'Q494511', 'Q158438', 'Q159675',
                     'Q559026', 'Q188860', 'Q28564', 'Q189867', 'Q95074', 'Q137773', 'Q778129', 'Q738570', 'Q50053',
                     'Q766277', 'Q2775969', 'Q634', 'Q1006876', 'Q5386', 'Q319141', 'Q3343298', 'Q173402', 'Q772547',
                     'Q5849', 'Q165194', 'Q131093', 'Q66016', 'Q128234', 'Q47053', 'Q1777138', 'Q55818', 'Q944816',
                     'Q46622', 'Q1758856', 'Q182659', 'Q189445', 'Q1110684', 'Q207338', 'Q728145', 'Q3032154', 'Q42523',
                     'Q210112', 'Q147538', 'Q963099', 'Q589183', 'Q41207', 'Q4407246', 'Q33881', 'Q131401', 'Q159979',
                     'Q211748', 'Q205020', 'Q902104', 'Q7270', 'Q473972', 'Q719487', 'Q697196', 'Q34627', 'Q178550',
                     'Q219261', 'Q35516', 'Q191992', 'Q183288', 'Q740445', 'Q7944', 'Q615150', 'Q1345234', 'Q3327874',
                     'Q45762', 'Q746369', 'Q178266', 'Q2679045', 'Q186386', 'Q134856', 'Q783930', 'Q86622', 'Q484692',
                     'Q841985', 'Q2919801', 'Q213369', 'Q182531', 'Q955824', 'Q838296', 'Q134447', 'Q83090', 'Q22721',
                     'Q1907114', 'Q1754117', 'Q1221156', 'Q2199', 'Q911663', 'Q221409', 'Q34749', 'Q74047', 'Q1958056',
                     'Q1070167', 'Q44559', 'Q379158', 'Q318', 'Q131212', 'Q37484', 'Q82480', 'Q39816', 'Q24706',
                     'Q101352', 'Q163323', 'Q161387', 'Q1138494', 'Q17888', 'Q158555', 'Q236036', 'Q264965', 'Q3624078',
                     'Q26529', 'Q56019', 'Q543151', 'Q182653', 'Q4182287', 'Q201658', 'Q708676', 'Q26398', 'Q3814081',
                     'Q2061186', 'Q6540832', 'Q1021711', 'Q185086', 'Q42032', 'Q630531', 'Q25400', 'Q146083', 'Q841753',
                     'Q427523', 'Q917146', 'Q620471', 'Q2592651', 'Q2635894', 'Q8092', 'Q173600', 'Q243249', 'Q1750705',
                     'Q2679157', 'Q690840', 'Q150093', 'Q283202', 'Q223393', 'Q48264', 'Q4618', 'Q12791', 'Q105420',
                     'Q2775236', 'Q2177636', 'Q187659', 'Q161376', 'Q427287', 'Q104157', 'Q50198', 'Q428875', 'Q261543',
                     'Q3700011', 'Q2713747', 'Q11184', 'Q1270588']

class EntityTypeProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_labels(self):
        """See base class."""
        #return trex_relations_et
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = sorted(line['labels'])

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


relations = ['per:siblings', 'per:parents', 'org:member_of', 'per:origin', 'per:alternate_names', 'per:date_of_death',
             'per:title', 'org:alternate_names', 'per:countries_of_residence', 'org:stateorprovince_of_headquarters',
             'per:city_of_death', 'per:schools_attended', 'per:employee_of', 'org:members', 'org:dissolved',
             'per:date_of_birth', 'org:number_of_employees/members', 'org:founded', 'org:founded_by',
             'org:political/religious_affiliation', 'org:website', 'org:top_members/employees', 'per:children',
             'per:cities_of_residence', 'per:cause_of_death', 'org:shareholders', 'per:age', 'per:religion',
             'no_relation',
             'org:parents', 'org:subsidiaries', 'per:country_of_birth', 'per:stateorprovince_of_death',
             'per:city_of_birth',
             'per:stateorprovinces_of_residence', 'org:country_of_headquarters', 'per:other_family',
             'per:stateorprovince_of_birth',
             'per:country_of_death', 'per:charges', 'org:city_of_headquarters', 'per:spouse']


class RelationClassificationProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), dataset_type)

    def get_labels(self):
        """See base class."""
        # return ["0", "1"]
        return relations

    def _create_examples(self, lines, dataset_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            # text_a: tokenized words
            text_a = line['text']
            # text_b: other information
            text_b = (line['ents'][0][1], line['ents'][0][2], line['ents'][1][1], line['ents'][1][2])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


def convert_to_t5_examples_trex(examples, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    t5_examples = []
    count = 0
    count_unk = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        inputs = ' '.join(example.text_a)
        subj_start, subj_end, obj_start, obj_end = example.text_b
        if example.label in relations:
            label_text = [relations[example.label]]
            labels = example.text_a[subj_start: subj_end + 1] + label_text + example.text_a[obj_start: obj_end + 1]
            labels = ' '.join(labels)
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input: %s" % inputs)
                logger.info("label: %s" % labels)
            count += 1
            t5_examples.append({'input': inputs, 'label': labels})
        else:
            count += 1

    print('The number of examples with known relation: {}'.format(count))
    print('The number of examples with unknown relation: {}'.format(count_unk))
    return t5_examples


def convert_to_t5_examples_et(examples, triplets=None, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    t5_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        start, end = example.text_b[0], example.text_b[1]
        text = example.text_a
        inputs = text[:start] + '<extra_id_0>' + text[start:end] +  '<extra_id_1>' + text[end:] 
        
        labels = example.label[0] if len(example.label) > 0 else ''
        i = 1
        while i < len(example.label):
            extra_tok = '<extra_id_{}>'.format(i+1)
            labels += extra_tok
            labels += example.label[i]
            i += 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input: %s" % inputs)
            logger.info("label: %s" % labels)
        t5_examples.append({'input': inputs, 'label': labels})

    return t5_examples


def convert_to_t5_examples_et_1(examples, triplets=None, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    t5_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        triplet_len = len(triplets[ex_index] + '<extra_id_0>')
        start, end = triplet_len + example.text_b[0], triplet_len + example.text_b[1]
        text = triplets[ex_index] + '<extra_id_0>' + example.text_a
        inputs = text[:start] + '<extra_id_1>' + text[start:end] +  '<extra_id_2>' + text[end:] 
        
        labels = example.label[0] if len(example.label) > 0 else ''
        i = 1
        while i < len(example.label):
            extra_tok = '<extra_id_{}>'.format(i+2)
            labels += extra_tok
            labels += example.label[i]
            i += 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input: %s" % inputs)
            logger.info("label: %s" % labels)
        t5_examples.append({'input': inputs, 'label': labels})

    return t5_examples


def convert_to_t5_examples_et_2(examples, triplets=None, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    t5_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        start, end = example.text_b[0], example.text_b[1]
        text = example.text_a
        inputs = text[:start] + '<@>' + text[start:end] +  '<#>' + text[end:] 
        
        labels = example.label[0] if len(example.label) > 0 else ''
        i = 1
        while i < len(example.label):
            labels += '<&>'
            labels += example.label[i]
            i += 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input: %s" % inputs)
            logger.info("label: %s" % labels)
        t5_examples.append({'input': inputs, 'label': labels})

    return t5_examples


def convert_to_t5_examples_et_3(examples, triplets=None, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    t5_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        start, end = example.text_b[0], example.text_b[1]
        text = example.text_a
        inputs = text[:start] + '<@>' + text[start:end] +  '<#>' + text[end:] 
        
        labels = '<@>' + text[start:end] +  '<#>'
        labels += example.label[0] if len(example.label) > 0 else ''
        i = 1
        while i < len(example.label):
            labels += '<&>'
            labels += example.label[i]
            i += 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input: %s" % inputs)
            logger.info("label: %s" % labels)
        t5_examples.append({'input': inputs, 'label': labels})

    return t5_examples


def convert_to_t5_examples_et_4(examples, triplets=None, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    t5_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        start, end = example.text_b[0], example.text_b[1]
        text = example.text_a
        inputs = text[:start] + '<extra_id_0>' + text[start:end] +  '<extra_id_1>' + text[end:] 
        
        labels = '<extra_id_0>' + text[start:end] +  '<extra_id_1>'
        labels += example.label[0] if len(example.label) > 0 else ''
        i = 1
        while i < len(example.label):
            extra_tok = '<extra_id_{}>'.format(i+1)
            labels += extra_tok
            labels += example.label[i]
            i += 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input: %s" % inputs)
            logger.info("label: %s" % labels)
        t5_examples.append({'input': inputs, 'label': labels})

    return t5_examples


def convert_to_t5_examples_rc(examples, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # label_map = {label: i for i, label in enumerate(label_list)}

    t5_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_a = example.text_a
        subj_start, subj_end, obj_start, obj_end = example.text_b
        labels = relations[example.label]
        
        if subj_start < obj_start:
            inputs = text_a[:subj_start] + '<extra_id_0>' + text_a[subj_start:subj_end] + '<extra_id_1>' + \
                text_a[subj_end:obj_start] + '<extra_id_2>' + text_a[obj_start:obj_end] + '<extra_id_3>' + \
                    text_a[obj_end:]
        else:
            inputs = text_a[:obj_start] + '<extra_id_2>' + text_a[obj_start:obj_end] + '<extra_id_3>' + \
                text_a[obj_end:subj_start] + '<extra_id_0>' + text_a[subj_start:subj_end] + '<extra_id_1>' + \
                    text_a[subj_end:]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input: %s" % inputs)
            logger.info("label: %s" % labels)
        t5_examples.append({'input': inputs, 'label': labels})

    return t5_examples


def convert_to_t5_examples_with_rc_1(examples, triplets, relations=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # label_map = {label: i for i, label in enumerate(label_list)}

    t5_examples = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_a = example.text_a
        triplet_len = len(triplets[ex_index] + '<extra_id_0>')
        subj_start, subj_end, obj_start, obj_end = [el+triplet_len for el in example.text_b]
        labels = relations[example.label]
        
        if subj_start < obj_start:
            inputs = triplets[ex_index] + '<extra_id_0>' + \
                text_a[:subj_start] + '<extra_id_1>' + text_a[subj_start:subj_end] + '<extra_id_2>' + \
                text_a[subj_end:obj_start] + '<extra_id_3>' + text_a[obj_start:obj_end] + '<extra_id_4>' + \
                    text_a[obj_end:]
        else:
            inputs = triplets[ex_index] + '<extra_id_0>' + \
                text_a[:obj_start] + '<extra_id_3>' + text_a[obj_start:obj_end] + '<extra_id_4>' + \
                text_a[obj_end:subj_start] + '<extra_id_1>' + text_a[subj_start:subj_end] + '<extra_id_2>' + \
                    text_a[subj_end:]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input: %s" % inputs)
            logger.info("label: %s" % labels)
        t5_examples.append({'input': inputs, 'label': labels})

    return t5_examples


processors = {
    "trex": TREXProcessor,
    "openentity": EntityTypeProcessor,
    "figer": EntityTypeProcessor,
    "tacred": RelationClassificationProcessor,
    "fewrel": RelationClassificationProcessor,
}

preprocessor_functions = {
    "trex": convert_to_t5_examples_trex,
    "openentity": (convert_to_t5_examples_et,
                    convert_to_t5_examples_et_1,
                        convert_to_t5_examples_et_2,
                            convert_to_t5_examples_et_3,
                                convert_to_t5_examples_et_4),
    "figer": (convert_to_t5_examples_et,
                    convert_to_t5_examples_et_1,
                        convert_to_t5_examples_et_2,
                            convert_to_t5_examples_et_3,
                                convert_to_t5_examples_et_4),
    "tacred": (convert_to_t5_examples_rc,
                    convert_to_t5_examples_rc_1)
    "fewrel": (convert_to_t5_examples_rc,
                    convert_to_t5_examples_rc)
}
