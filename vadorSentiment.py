from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# 测试原始的VaderSentiment词典，它不会识别"Fxxk"这一单词
sentences = [
    "fxxk",
    "SUX",
    "nigger",
    "Love from China",
    "I love Chipotle, but the food isn't always consistent at this location (nor is it anywhere else, but it's still worth noting at this one in particular). For the 2nd visit in a row, I've watched 2 different employees dump the black beans onto my burrito without draining them first! Having a soggy burrito dripping with bean juice is not my idea of quality. Another issue I have is that the chips are often prepared rather poorly. During my past few visits, I've gotten bags of chips that aren't fully fried! The softness and sogginess of the chips isn't helped by the overload of salt that they've drowned the chips in after prematurely pulling them from the fryer. Also, not to sound like a silly racist, but where have all the Mexicans gone that used to work at this location? Perhaps I've just gone when they aren't on shift. As far as I could tell, the food quality was always above average when it wasn't all white people working there (again, not to sound like a pathetic race-obsessed honky. I know this is Louisville). Overall I really can't complain, but just be weary when ordering chips! Make Chipotle Mexican Again!"
]
print('使用原始词典')
analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print(sentence)
    print(str(vs))

# 修改原始词典，加入FXXK
new_words = {
    'fxxk': -2.0,
}

print('\n使用全新词典')
analyzer.lexicon.update(new_words)
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print(sentence)
    print(str(vs))

# 读取dta文件, 记得修改路径到你的文件夹
df = pd.read_stata('C:/Users/95328/OneDrive - HKUST Connect/racism/discrimination text mining.dta')
test = df.head(100)
test_texts = test['text']
print('\n测试数据集中前100行')
for sentence in test_texts:
    vs = analyzer.polarity_scores(sentence)
    print(sentence)
    print(str(vs))