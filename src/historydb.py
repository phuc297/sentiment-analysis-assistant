import sqlite3


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class HistoryDB(metaclass=SingletonMeta):

    def __init__(self, db_path="history.db"):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.cursor = self.con.cursor()
        self._initialized = True
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence TEXT,
                predicted_label TEXT,
                negative_score FLOAT,
                positive_score FLOAT,
                neutral_score FLOAT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.con.commit()

    def add(self, record):
        sql_query = """
            INSERT INTO history (
                sentence, predicted_label, negative_score, positive_score, neutral_score, timestamp
            ) VALUES (
                :sentence, :predicted_label, :negative_score, :positive_score, :neutral_score, :timestamp
            )
            """
        self.cursor.execute(sql_query, record)
        self.con.commit()

    def get(self, id):
        sql_query = """
            SELECT id, sentence, predicted_label, negative_score, positive_score, 
                neutral_score, STRFTIME('%d/%m/%Y, %I:%M %p', timestamp) AS timestamp
            FROM history WHERE id = ?
            """
        self.cursor.execute(sql_query, (id,))
        result_list = dict(self.cursor.fetchall()[0])
        result_list['prob_dict'] = {
            "negative": result_list['negative_score'],
            "positive": result_list['positive_score'],
            "neutral": result_list['neutral_score'],
        }
        return result_list

    def get_all(self):
        sql_query = """
            SELECT id, sentence, predicted_label, negative_score, positive_score, 
                neutral_score, STRFTIME('%d/%m/%Y, %I:%M %p', timestamp) AS timestamp
            FROM history ORDER BY id DESC
            """
        self.cursor.execute(sql_query)
        result_list = self.cursor.fetchall()
        return [dict(s) for s in result_list]

    def save(self, input_text, label, probabilities: list, timestamp):
        record = create_record(
            input_text, label, probabilities, timestamp)
        self.add(record)


def create_record(sentence, label, probabilities, timestamp):
    return {
        'sentence': sentence,
        'predicted_label': label,
        'negative_score': probabilities['negative'],
        'positive_score': probabilities['positive'],
        'neutral_score': probabilities['neutral'],
        'timestamp': timestamp,
    }


if __name__ == "__main__":
    db = HistoryDB()
    # cols_name = ['sentence', 'predicted_label', 'negative_score',
    #              'positive_score', 'neutral_score', 'timestamp']
    print(db.get(2))

    # for s in db.get_all():
    #     print(s)

    # from datetime import datetime
    # for s in db.get_all()[:1]:
    #     dt_string = s['timestamp']

    #     dt_object = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S.%f")
    #     new_format_dtstring = dt_object.strftime("%d/%m/%Y, %I:%M %p")
    #     print(f"Format 1: {new_format_dtstring}")
