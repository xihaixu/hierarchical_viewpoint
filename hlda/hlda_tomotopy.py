import csv
import tomotopy as tp
# 测试该组件可运行的模式
# print(tp.isa) # prints 'avx2', 'avx', 'sse2' or 'none'

mdl = tp.HLDAModel(depth=3, min_cf=100)
csvfile = open('1.7-1.21\\filter_for_hcq.csv', encoding = 'utf-8')
data = csv.reader(csvfile)
ix = 0
for line in data:
    ix += 1
    if ix>1:
        text = line[6].strip().lower().split()  #3,6:phrase; 4,7:words
        print(text)
        if len(text)>0:
            mdl.add_doc(text)

print('Training model by iterating over the corpus 100 times, 10 iterations at a time')
iterations = 10
for i in range(0, 100, iterations):
    mdl.train(iterations)
    print('Iteration: #{}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

for k in range(mdl.k):
    if not mdl.is_live_topic(k):
        continue

    print('child of topic #%s - Level: %r, number of documents:%r' % (mdl.parent_topic(k), mdl.level(k), mdl.num_docs_of_topic(k)) )
    # print(mdl.parent_topic(k))
    print('Top 10 words of global topic #{}'.format(k))
    print(mdl.get_topic_words(k, top_n=15))

# parent_topics = [k for k in range(mdl.k) if mdl.children_topics(k) and mdl.num_docs_of_topic(parent_topic) > 100]
# for parent_topic in parent_topics:
#     child_topics = [child_topic for child_topic in mdl.children_topics(parent_topic) if
#                     mdl.num_docs_of_topic(child_topic) > 100]
#     if child_topics:
#         print('\n\n')
#     print('Top 10 words of level %s parent topic #%s of %s documents: %r' % (
#     mdl.level(parent_topic), parent_topic, mdl.num_docs_of_topic(parent_topic),
#     mdl.get_topic_words(parent_topic, top_n=10)))
#
#     for child_topic in child_topics:
#         print('    Top 10 words of child topic #%s: %r' % (child_topic, mdl.get_topic_words(child_topic, top_n=10)))