import numpy as np
import pandas as pd

# mail: angelo.porrello@unimore.it

def es1():
    # 1. Importa pandas come pd e controlla la versione
    pass


def es2():
    # 2. Crea una pandas series da: una lista, un numpy array, un dizionario
    # Input:
    import numpy as np

    mylist = list('abcedfghijklmnopqrstuvwxyz')
    myarr = np.arange(26)
    mydict = dict(zip(mylist, myarr))
    #
    pass


def es3():
    # 3. Converti la series ser in un dataframe
    # Input:
    mylist = list('abcedfghijklmnopqrstuvwxyz')
    myarr = np.arange(26)
    mydict = dict(zip(mylist, myarr))
    ser = pd.Series(mydict)
    #
    pass


def es4():
    # 4. Combina ser1 e ser2 per formare un dataframe
    # Input:
    import numpy as np

    ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
    ser2 = pd.Series(np.arange(26))
    #
    pass


def es5():
    # 5. Riforma (reshape) la series ser in un dataframe con 7 righe e 5 colonne
    # Input:
    ser = pd.Series(np.random.randint(1, 10, 35))
    #
    pass


def es6():
    # 6. Trova le posizioni dei numeri che sono multipli di 3 in ser
    # Input:
    ser = pd.Series(np.random.randint(1, 10, 7))
    #
    pass


def es7():
    # 7. Impila ser1 e ser2 verticalmente e orizzontalmente
    # Input:
    ser1 = pd.Series(range(5))
    ser2 = pd.Series(list('abcde'))
    # Vertical
    pass


def es8():
    # 8. Converti una series di date in forma testuale in oggetti datetime
    # Input:
    ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
    # Output:
    # 0   2010-01-01 00:00:00
    # 1   2011-02-02 00:00:00
    # 2   2012-03-03 00:00:00
    # 3   2013-04-04 00:00:00
    # 4   2014-05-05 00:00:00
    # 5   2015-06-06 12:20:00
    # dtype: datetime64[ns]
    pass


def es9():
    # 9. Calcola la media dei pesi di ogni frutto (i.e., ottieni la media di una series raggruppata per un'altra series)
    # Input:
    fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
    weights = pd.Series(np.linspace(1, 10, 10))
    print(weights.tolist())
    print(fruit.tolist())
    pass


def es10():
    # 10. Importa come dataframe solo le colonne 'crim' e 'medv' del dataset BostonHousing
    # Input path:
    # 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
                     usecols=['crim', 'medv'])


def es11():
    # 11. Ricava il numero di righe e colonne, i datatype e le statistiche riassuntive di ciascuna colonna di df
    # Input:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
    #
    pass


def es12():
    # 12. Rinomina la colonna Type come CarType in df
    # Input:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
    # print(df.columns)
    # > Index(['Manufacturer', 'Model', 'Type', 'Min.Price', 'Price', 'Max.Price', 'MPG.city', 'MPG.highway',
    # >        'AirBags', 'DriveTrain', 'Cylinders', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile',
    # >        'Man.trans.avail', 'Fuel.tank.capacity', 'Passengers', 'Length', 'Wheelbase', 'Width', 'Turn.circle',
    # >	       'Rear.seat.room', 'Luggage.room', 'Weight', 'Origin', 'Make'], dtype='object')
    # Output:
    # print(df.columns)
    # > Index(['Manufacturer', 'Model', 'CarType', 'Min_Price', 'Price', 'Max_Price', 'MPG.city', 'MPG.highway',
    # >	       'AirBags', 'DriveTrain', 'Cylinders', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile',
    # > 	   'Man.trans.avail', 'Fuel.tank.capacity', 'Passengers', 'Length', 'Wheelbase', 'Width', 'Turn.circle',
    # >	       'Rear.seat.room', 'Luggage.room', 'Weight', 'Origin', 'Make'], dtype='object')
    #
    pass


def es13():
    # 13. Controlla se ci sono valori mancanti in df
    # Input:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
    #
    pass


def es14():
    # 14. Sostituisci i valori mancanti nelle colonne Min.Price e Max.Price con le loro rispettive medie
    # Input:
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
    #
    pass


def es15():
    # 15. Determina le posizioni in cui i valori di due colonne corrispondono
    # Input:
    df = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),
                       'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})
    #
    pass
