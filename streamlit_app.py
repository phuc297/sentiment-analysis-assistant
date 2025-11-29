import streamlit as st
import requests
import config
import pandas as pd
from datetime import datetime
from historydb import HistoryDB, create_record

LABELS = {"negative": "TIÊU CỰC (Negative)",
          "positive": "TÍCH CỰC (Positive)",
          "neutral": "TRUNG TÍNH (Neutral)",
          }


if 'history' not in st.session_state:
    st.session_state.history = None

if "last_predicted" not in st.session_state:
    st.session_state.last_predicted = {"sentence": None,
                                       "timestamp": None,
                                       }


st.title("Sentiment Analysis cho Tiếng Việt")
st.caption(f"Mô hình sử dụng: {config.MODEL_NAME}")

predict_tab, history_tab = st.tabs(["📈 Dự đoán", "🗃 Lịch sử"])

with predict_tab:
    if st.session_state.history is not None:
        value_input = st.session_state.history['sentence']
        input_text = st.text_area("Nhập câu tiếng Việt:", value=value_input)
    else:
        input_text = st.text_area("Nhập câu tiếng Việt:")

    predict_btn = st.button("Dự đoán Cảm xúc")


@st.cache_data
def predict_sentiment(sentence):
    data = {"text": sentence}
    response_data = requests.post(config.SA_API_URL, json=data).json()
    return response_data['label'], response_data['probabilities']


@st.cache_resource
def get_database() -> HistoryDB:
    db = HistoryDB()
    return db


history_db = get_database()


def onclick_load_history(id=None):
    st.session_state.history = history_db.get(id)


def render_history_sidebar():
    with st.sidebar:
        st.markdown("## Lịch Sử Dự Đoán")
        for i, s in enumerate(history_db.get_all()):
            btn = st.button(f"{s['sentence']}", on_click=onclick_load_history,
                            args=[s['id']], width="stretch",
                            key=f"hist_btn_{str(i)}")


def render_results(label, probabilities, timestamp):
    with predict_tab:
        st.markdown("## Kết quả Dự đoán")

        if "positive" in label:
            st.success(f"**Dự đoán Cảm xúc:** {LABELS[label]}")
        elif "negative" in label:
            st.error(f"**Dự đoán Cảm xúc:** {LABELS[label]}")
        else:
            st.info(f"**Dự đoán Cảm xúc:** {LABELS[label]}")

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(LABELS["negative"],
                      f"{probabilities["negative"]}%", border=True)
        with col2:
            st.metric(LABELS["neutral"],
                      f"{probabilities["neutral"]}%", border=True)
        with col3:
            st.metric(LABELS["positive"],
                      f"{probabilities["positive"]}%", border=True)

        st.markdown(f"Thời gian: **{timestamp}**")


def save_history(input_text, label, probabilities, timestamp):
    record = create_record(
        input_text, label, probabilities, timestamp)
    history_db.add(record)


def prediction_flow():
    if not input_text.strip():
        st.warning("Vui lòng nhập văn bản để dự đoán.")
        return

    if len(input_text.split()) < 2:
        st.warning("Câu quá ngắn.")
        return

    label, probabilities = predict_sentiment(input_text)
    if not label:
        return

    if input_text != st.session_state.last_predicted['sentence']:
        timestamp_dt = datetime.now()
        timestamp = timestamp_dt.strftime("%d/%m/%Y, %I:%M %p")
        save_history(input_text, label, probabilities, timestamp_dt)
        st.session_state.last_predicted = {"sentence": input_text,
                                           "timestamp": timestamp,
                                           }
    else:
        timestamp = st.session_state.last_predicted["timestamp"]

    render_results(label, probabilities, timestamp)


def render_history():
    label = st.session_state.history['predicted_label']
    probabilities = st.session_state.history['prob_dict']
    timestamp = st.session_state.history['timestamp']

    st.session_state.last_predicted = {"sentence": input_text,
                                       "timestamp": timestamp,
                                       }

    render_results(label, probabilities, timestamp)
    st.session_state.history = None


if predict_btn:
    prediction_flow()
if st.session_state.history is not None:
    render_history()

render_history_sidebar()
with history_tab:
    hist_list = history_db.get_all()
    if hist_list:
        hist_table = pd.DataFrame(hist_list).drop(columns='id')
        st.dataframe(hist_table, hide_index=True)
