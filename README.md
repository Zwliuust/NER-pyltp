# NER-pyltp
分析短文本提取出其中的人名地名。<br>
调用pyltp中命名实体识别功能，初步检出人名地名。在此基础上，添加规则优化效果，检出更详细的地名（详细到门牌号）、不检出大地名（不够敏感，不需要检出）以及将有相同字符串的地名合并。
