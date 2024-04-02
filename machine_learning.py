import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#lichess platformunda oynanan bazı satranç maçlarının kayıtlarını içeren verisetini okuyoruz. 
df = pd.read_csv('Veriseti/games.csv')
#transform uygulayabileceğimiz ordinal encoder'ı tanımlıyoruz.
oe = OrdinalEncoder() 
#id'ye karşılık gelen açılış ismini belirtebilmek adına ilgili verisetine kolon ekliyoruz.
df.insert(0, 'ID', range(0, 0 + len(df)))
#merge edeceğimiz kolonları belirliyoruz.
colsforMerge = ['ID','opening_name']
#merge edeceğimiz kolonların seçili olduğu seti oluşturuyoruz.
merged = df[colsforMerge]
#oluşan seti transform edip uygun formata geçiriyoruz.
df[colsforMerge] = oe.fit_transform(df[colsforMerge]) 
#id üzerinden birleştirip açılış isim ve numaralarını listelemek içinset oluşturuyoruz.
merged_last = pd.merge(merged, df[colsforMerge], on='ID', how='left')
#uygun şekilde kolon isimlendirmesi yapıyoruz.
merged_last = merged_last.rename(columns={'opening_name_y': 'opening_number', 'opening_name_x': 'opening_name'})
#listelenecek kolonları belirliyoruz.
mergeShowColumns = ['opening_name','opening_number']
#açılış ismine istinaden seçilebilecek açılış numaralarını listeliyoruz.
print(merged_last[mergeShowColumns].drop_duplicates())


#yapılacak bir satranç maçı için rating'lere göre hangi tarafın kazanma şansının kaç
#olduğu bilgisini öğrenebilmek adına makinayı bu bilgilerle eğitip sonrasında doğruluğu
#için tespiti ve rating'lere göre yapılacak örnek bir maç için test case'leri edinebilmek
#adına beyaz ve siyahın rating bilgilerini içeren kolonları belirliyoruz.
cols = ['white_rating', 'black_rating']
#x değeri için sette bu kolonları içeren alt seti seçiyoruz.
X = df[cols]
#y değeri için kazanın bilgisini tutan winner kolonunu içeren seti belirliyoruz.
y = df['winner']
#eğitim işlemleri için düzenlemeleri gerçekleştiriyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
#uygulayacağımız regresyon değişkenini tanımlıyoruz.
lr = LogisticRegression()
#eğitim işlemini gerçekleştiriyoruz.
lr.fit(X_train, y_train)
#score ile eğitimimizin başarısını ölçüyoruz.
lr.score(X_test, y_test)

#Konsolda düzen için ayırıyoruz.
print('--------------------')

#Logistic Regression işlemlerini gerçekleştireceğiz. Konsola bunu belirten ibare ekliyoruz.
print('Regression işlemleri')
#konsolda aralık bırakıyoruz.
print()

#ÖRNEK - 1
#siyahın rating'inin biraz daha fazla olduğu bir örnek hazırlıyoruz.
test = np.array([1786, 1800])
#test düzenini ayarlıyoruz.
test = test.reshape(1, -1)
#test için kullanıyor olduğumuz beyazın rating'ini yazdırıyoruz.
print('white rating : 1786')
#test için kullanıyor olduğumuz siyahın rating'ini yazdırıyoruz.
print('black rating : 1800')
#testin gerçekleştirildiği tarafı tahmin ediyoruz (white/black).
print(lr.predict(test) + ' possible win rate:')
#kazanma ihtimalini tahminliyoruz.
print(max(lr.predict_proba(test)[0]))
#konsolda aralık bırakıyoruz.
print()

#ÖRNEK - 2
#beyazın çok daha fazla rating'e sahip olduğu bir eşleşme testi için örnek hazırlıyoruz.
test = np.array([2300, 1996])
#test düzenini ayarlıyoruz.
test = test.reshape(1, -1)
#test için kullanıyor olduğumuz beyazın rating'ini yazdırıyoruz.
print('white rating : 2300')
#test için kullanıyor olduğumuz siyahın rating'ini yazdırıyoruz.
print('black rating : 1996')
#tahmini değişken üzerinde tutuyoruz.
#testin gerçekleştirildiği tarafı tahmin ediyoruz (white/black).
print(lr.predict(test) + ' possible win rate:')
#kazanma ihtimalini tahminliyoruz.
print(max(lr.predict_proba(test)[0]))
#konsolda aralık bırakıyoruz.
print()
#10-fold matirisi için yazıyoruz.
print('10-fold matrisi')
#10-fold uygulanıyor.
print(cross_val_score(lr, X_test, y_test, cv=10))
#model skoru çıkartılıyor.
scores = cross_val_score(lr, X, y, scoring='accuracy', cv=10, n_jobs=-1)
#10-fol ile başarı değeri yazdırılıyor.
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
#karmaşıklık matrisini yazdırmak için hazırlıyoruz.
print('Confusion Matrix: ')
#karmaşıklık matrisini çıkartıyoruz.
print(confusion_matrix(y_test, lr.predict(X_test)))
#konsolda aralık bırakıyoruz.
print()

#Konsolda düzen için ayırıyoruz.
print('--------------------')

#Naive Bayes işlemlerini gerçekleştireceğiz. Konsola bunu belirten ibare ekliyoruz.
print('Naive Bayes işlemleri')
#konsolda aralık bırakıyoruz.
print()
#Naive Bayes işlemi için eğitim ve test verilerini ayarlıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#scaler'ı tanımlıyoruz.
sc_X = StandardScaler()
#eğitim verisi için scaling işlemini uyguluyoruz.
X_train = sc_X.fit_transform(X_train)
#test verisi için scaling işlemini uyguluyoruz.
X_test = sc_X.transform(X_test)
#naive bayes için değişkeni tanımlıyoruz.
classifierNaive = GaussianNB()
#eğitim işlemini gerçekleştiriyoruz.
classifierNaive.fit(X_train, y_train)

#ÖRNEK - 3
#siyahın rating'inin biraz daha fazla olduğu bir örnek hazırlıyoruz.
test = np.array([1786, 1800])
#test düzenini ayarlıyoruz.
test = test.reshape(1, -1)
#test için kullanıyor olduğumuz beyazın rating'ini yazdırıyoruz.
print('white rating : 1786')
#test için kullanıyor olduğumuz siyahın rating'ini yazdırıyoruz.
print('black rating : 1800')
#testin gerçekleştirildiği tarafı tahmin ediyoruz (white/black).
print(classifierNaive.predict(test))
#kazanma ihtimalini tahminliyoruz.
print(max(classifierNaive.predict_proba(test)[0]))
#konsolda aralık bırakıyoruz.
print()

#ÖRNEK - 4
#beyazın çok daha fazla rating'e sahip olduğu bir eşleşme testi için örnek hazırlıyoruz.
test = np.array([2300, 1996])
#test düzenini ayarlıyoruz.
test = test.reshape(1, -1)
#test için kullanıyor olduğumuz beyazın rating'ini yazdırıyoruz.
print('white rating : 2300')
#test için kullanıyor olduğumuz siyahın rating'ini yazdırıyoruz.
print('black rating : 1996')
#testin gerçekleştirildiği tarafı tahmin ediyoruz (white/black).
print(classifierNaive.predict(test))
#kazanma ihtimalini tahminliyoruz.
print(max(classifierNaive.predict_proba(test)[0]))
#konsolda aralık bırakıyoruz.
print()
#10-fold matirisi için yazıyoruz.
print('10-fold matrisi')
#10-fold uygulanıyor.
print(cross_val_score(classifierNaive, X_test, y_test, cv=10))
#model skoru çıkartılıyor.
scores = cross_val_score(classifierNaive, X, y, scoring='accuracy', cv=10, n_jobs=-1)
#10-fol ile başarı değeri yazdırılıyor.
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
#karmaşıklık matrisini yazdırmak için hazırlıyoruz.
print('Confusion Matrix: ')
#karmaşıklık matrisini çıkartıyoruz.
print(confusion_matrix(y_test, classifierNaive.predict(X_test)))
#konsolda aralık bırakıyoruz.
print()

#Konsolda düzen için ayırıyoruz.
print('--------------------')

#K-NN işlemlerini gerçekleştireceğiz. Konsola bunu belirten ibare ekliyoruz.
print('K-NN işlemleri')

#oynayacağımız açılışa göre kazanma şansımızın ne kadar olduğunu belirleyeceğimiz bir
#tahminleme için açılış bilgisini içeren kolonu belirliyoruz.
cols = ['opening_name']
#seçilen kolona istinaden set oluşturuyoruz.
merged = df[cols]
#ilgili kolonu içeren seti transform edip uygun formata çeviriyoruz.
df[cols] = oe.fit_transform(df[cols]) 
#x değerimiz için açılış bilgisini içeren seti belirliyoruz.
X = df[cols]
#y değeri için kazanan bilgisini içeren winner kolonunu içeren seti belirliyoruz.
y = df['winner']
#eğitim ve test verisi için bilgileri ayarlıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
#uygulayacağımız knn için değişkeni ayarlıyoruz.
classifier = KNeighborsClassifier(n_neighbors=5)
#eğitim verileri ile knn'i gerçekleştiriyoruz.
classifier.fit(X_train, y_train)
#score ile eğitimimizin başarısını ölçüyoruz.
classifier.score(X_test, y_test)


#ÖRNEK - 5
#Örnek olarak Slav Defense: Exchange Variation açılışına karşılık gelen değeri test verimiz
#için hazırlıyoruz.
test = np.array([1387])
#test düzenini ayarlıyoruz.
test = test.reshape(1, -1)
#test için kullanıyor olduğumuz açılış ismini yazdırıyoruz.
print('opening : Slav Defense: Exchange Variation')
#testin gerçekleştirildiği tarafı tahmin ediyoruz (white/black).
print(classifier.predict(test) + ' possible win rate with chosen opening:')
#ilgili açılışı oynayarak kazanma ihtimalini tahminliyoruz.
print(max(classifier.predict_proba(test)[0]))
#10-fold matirisi için yazıyoruz.
print('10-fold matrisi')
#10-fold uygulanıyor.
print(cross_val_score(classifier, X_test, y_test, cv=10))
#model skoru çıkartılıyor.
scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=10, n_jobs=-1)
#10-fol ile başarı değeri yazdırılıyor.
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
#karmaşıklık matrisini yazdırmak için hazırlıyoruz.
print('Confusion Matrix: ')
#karmaşıklık matrisini çıkartıyoruz.
print(confusion_matrix(y_test, classifier.predict(X_test)))