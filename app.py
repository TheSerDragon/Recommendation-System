import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main_page():
    st.title('Главная страница')
    st.image('https://w.forfun.com/fetch/e2/e29735d2e532b4aeb8ee2417816fa776.jpeg')
    st.write('Добро пожаловать в инновационную платформу, где вы можете открыть мир видеоигр с новой перспективы. Мы предлагаем уникальную рекомендательную систему, специально разработанную для того, чтобы помочь вам развивать профессиональные навыки через игровой опыт.')
    st.write('С помощью нашей рекомендательной системы вы сможете находить игры, наиболее подходящие для вашего профессионального роста. Независимо от того, хотите ли вы улучшить лидерские качества, развить стратегическое мышление, улучшить коммуникативные навыки или освоить новые технологии – у нас есть подходящая игра для вас.')

    st.header('Функционал:')
    st.markdown('- Получите персонализированные рекомендации.')

def video_game_page():
    df = load_data("data/all_data.csv")
    st.title('Поиск видеоигры')
    
    # Добавляем элементы интерфейса для выбора критериев поиска
    st.subheader('Выберите критерии поиска:')
    search_criteria = st.multiselect('Выберите критерии:', ['Название', 'Жанр', 'Платформа'])

    search_query = {}
    for criteria in search_criteria:
        if criteria == 'Название':
            search_query['Название'] = st.text_input('Введите название видеоигры:')
        elif criteria == 'Жанр':
            search_query['Жанр'] = st.text_input('Введите жанр видеоигры:')
        elif criteria == 'Платформа':
            platform_options = df['Platform'].unique().tolist()
            search_query['Платформа'] = st.selectbox('Выберите платформу:', platform_options)

    search_button = st.button('Поиск')

    if search_button:
        st.write('Вы ищете:')
        for key, value in search_query.items():
            st.write(f"{key}: {value}")

        # Фильтрация данных в зависимости от выбранных критериев поиска
        filtered_df = df.copy()
        for key, value in search_query.items():
            if key == 'Название':
                filtered_df = filtered_df[filtered_df['Title'].str.contains(value, case=False)]
            elif key == 'Жанр':
                filtered_df = filtered_df[filtered_df['Genre'].str.lower().str.contains(value.lower())]
            elif key == 'Платформа':
                filtered_df = filtered_df[filtered_df['Platform'] == value]

        if not filtered_df.empty:
            st.write('Найденные версии:')
            for index, row in filtered_df.iterrows():
                # Отображение краткой информации о версии игры в виде раскрывающегося блока
                with st.expander(f"{row['Title']} - {row['Genre']} ({row['Platform']})", expanded=False):
                    for column, value in row.drop('ID').items():  # исключаем столбец с ID
                        st.write(f"{column}: {value}")
        else:
            st.write('Видеоигры по вашему запросу не найдены')

def about_page():
    st.title('О нас')
    st.write('Это рекомендательная система, разработанная для помощи в получении персонализированных рекомендаций по различным товарам/фильмам/книгам.')
    st.write('Система использует алгоритмы машинного обучения для анализа предпочтений пользователя и предоставления соответствующих рекомендаций.')

def load_data(data):
    df = pd.read_csv(data)
    return df

# def vectorize_text_to_cosine_mat(data):
#    count_vect = CountVectorizer()
#    cv_mat = count_vect.fit_transform(data)
#    cosine_sim_mat = cosine_similarity(cv_mat)
#    return cosine_sim_mat

#def get_recommendation(title, cosine_sim_mat, df, num_of_rec = 10):
#    game_indices = pd.Series(df.index, index = df['Title']).drop_duplicates()
#    idx = game_indices[title]
#    sim_scores = list(enumerate(cosine_sim_mat[idx]))
#    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#    return sim_scores[1:]


def list_games_page():
    df = load_data("data/all_data.csv")
    st.title('Список доступных видеоигр')
    st.write('Вот список всех доступных видеоигр:')
    
    total_pages = len(df) // 10 + (1 if len(df) % 10 > 0 else 0)

    page_number = st.number_input('Выберите номер страницы:', min_value=1, max_value=total_pages, value=1)

    start_index = (page_number - 1) * 10
    end_index = min(start_index + 10, len(df))

    if start_index >= end_index:
        st.write('На этой странице нет игр.')

    for index in range(start_index, end_index):
        with st.expander(f"{df.iloc[index]['Title']} - {df.iloc[index]['Genre']} ({df.iloc[index]['Platform']})", expanded=False):
            for column, value in df.iloc[index].drop('ID').items():  # исключаем столбец с ID
                st.write(f"{column}: {value}")

def recommendation_page():
    df = load_data("data/all_data.csv")
    #cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
    st.title('Получить рекомендации')
    st.write('Введите название видеоигры, чтобы получить рекомендации похожих игр:')

    search_query = st.text_input('Введите название видеоигры:')
    search_button = st.button('Поиск')

    # if search_button:
        # Получаем рекомендации
    #    recommendations = get_recommendation(search_query, cosine_sim_mat, df)
    #    if recommendations:
    #        st.write('Рекомендации:')
    #        for idx, sim_score in recommendations:
    #            game_title = df.iloc[idx]['Title']
    #            st.write(f"{game_title} (Коэффициент сходства: {sim_score})")
    #    else:
    ##        st.write('По вашему запросу рекомендации не найдены.')

def main():
    selected_page = st.sidebar.radio('Выберите страницу', ['Главная страница', 'Список видеоигр', 'Поиск видеоигры', 'Рекомендации по видеоигре', 'О нас'])

    if selected_page == 'Главная страница':
        main_page()
    elif selected_page == 'Список видеоигр':
        list_games_page()
    elif selected_page == 'Поиск видеоигры':
        video_game_page()
    elif selected_page == 'Рекомендации по видеоигре':
        recommendation_page()
    elif selected_page == 'О нас':
        about_page()

if __name__ == "__main__":
    main()
