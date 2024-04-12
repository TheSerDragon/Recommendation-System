import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Отображение главной страницы
def main_page():
    st.title('Главная страница')
    st.image('https://w.forfun.com/fetch/e2/e29735d2e532b4aeb8ee2417816fa776.jpeg')
    st.write(
        'Добро пожаловать в инновационную платформу, где вы можете открыть мир видеоигр с новой перспективы. Мы предлагаем уникальную рекомендательную систему, специально разработанную для того, чтобы помочь вам развивать профессиональные навыки через игровой опыт.')
    st.write(
        'С помощью нашей рекомендательной системы вы сможете находить игры, наиболее подходящие для вашего профессионального роста. Независимо от того, хотите ли вы улучшить лидерские качества, развить стратегическое мышление, улучшить коммуникативные навыки или освоить новые технологии – у нас есть подходящая игра для вас.')

    st.header('Функционал:')
    st.markdown('- Получите персонализированные рекомендации.')
    st.markdown('- Ищите видеоигры по различным критериям.')
    st.markdown('- Просматривайте список доступных видеоигр.')


# Отображение страницы поиска видеоигры
def video_game_page():
    df = load_data("data/new_all_data.csv")
    st.title('Поиск видеоигры')

    platform_options = df['Platform'].unique().tolist()
    platform_options = [platform for platform in platform_options if platform != 'not specified']

    # Добавление элементов интерфейса для выбора критериев поиска
    st.write('Выберите критерии поиска:')
    search_criteria = st.multiselect('Выберите критерии:', ['Название', 'Жанр', 'Платформа', 'Оценка критиков', 'Оценка пользователей'])

    search_query = {}
    for criteria in search_criteria:
        if criteria == 'Название':
            title_options = df['Title'].unique().tolist()
            search_query['Название'] = st.selectbox('Введите название видеоигры:', title_options)
        elif criteria == 'Жанр':
            search_query['Жанр'] = st.text_input('Введите жанр видеоигры:')
        elif criteria == 'Платформа':
            search_query['Платформа'] = st.selectbox('Выберите платформу:', platform_options)
        elif criteria == 'Оценка критиков':
            min_critic_score, max_critic_score = st.slider('Выберите диапазон оценки критиков:', min_value=0, max_value=100, value=(0, 100), step=1)
            search_query['Оценка критиков'] = (min_critic_score, max_critic_score)
        elif criteria == 'Оценка пользователей':
            min_user_score, max_user_score = st.slider('Выберите диапазон оценки пользователей:', min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)
            search_query['Оценка пользователей'] = (min_user_score, max_user_score)

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
            elif key == 'Оценка критиков':
                min_score, max_score = value
                filtered_df['Metascore'] = pd.to_numeric(filtered_df['Metascore'], errors='coerce')
                filtered_df = filtered_df[(filtered_df['Metascore'] >= min_score) & (filtered_df['Metascore'] <= max_score)]
            elif key == 'Оценка пользователей':
                min_score, max_score = value
                filtered_df['Avg_Userscore'] = pd.to_numeric(filtered_df['Avg_Userscore'], errors='coerce')
                filtered_df = filtered_df[(filtered_df['Avg_Userscore'] >= min_score) & (filtered_df['Avg_Userscore'] <= max_score)]

        if not filtered_df.empty:
            st.write('Найденные версии:')
            for index, row in filtered_df.iterrows():
                # Отображение краткой информации о версии видеоигры в виде раскрывающегося блока
                with st.expander(f"{row['Title']} - {row['Genre']} ({row['Platform']})", expanded=False):
                    for column, value in row.drop('ID').items():  # исключаем столбец с ID
                        st.write(f"{column}: {value}")
        else:
            st.write('Видеоигры по вашему запросу не найдены')


# Отображение страницы с информацией о разработчике
def about_page():
    st.title('Разработчик')
    st.write('Студент 4 курса')
    st.write('МГТУ им. Н. Э. Баумана')
    st.write('Факультета ИУ "Информатика и системы управления"')
    st.write('Кафедры ИУ5 "Системы обработки информации и управления"')
    st.write('Группы ИУ5-81Б')
    st.write('Кондрахин Сергей Сергеевич')
    st.write('Контакты: kondrakhin.sergey@mail.ru')


# Загрузка данных из CSV-файла
@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df


# Векторизация жанров и создание матрицы косинусной схожести
@st.cache_data
def vectorize_genre_to_cosine_mat(data):
    count_vect = CountVectorizer(encoding='utf-8')
    vectors = count_vect.fit_transform(data.values.astype('U'))
    cosine_sim_mat = cosine_similarity(vectors)
    return cosine_sim_mat


# Функция для рекомендации похожих игр на основе жанров
@st.cache_data
def recommend(data, title, similarity):
    if title in data['title'].values:
        game_index = data[data['title'] == title].index[0]
        distances = similarity[game_index]
        game_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:16]
        similar_games = sorted(list(zip(data['title'].iloc[[index for index, _ in game_list]], distances)),
                               key=lambda x: x[1], reverse=True)[0:15]
        return similar_games
    else:
        return "Game not found in the dataset"


# Отображение списка доступных видеоигр
def list_games_page():
    df = load_data("data/new_all_data.csv")
    st.title('Список доступных видеоигр')
    st.write('Вот список всех доступных видеоигр:')

    total_pages = len(df) // 10 + (1 if len(df) % 10 > 0 else 0)

    page_number = st.number_input('Выберите номер страницы (1-501):', min_value=1, max_value=total_pages, value=1)

    start_index = (page_number - 1) * 10
    end_index = min(start_index + 10, len(df))

    if start_index >= end_index:
        st.write('На этой странице нет игр.')

    for index in range(start_index, end_index):
        with st.expander(f"{df.iloc[index]['Title']} - {df.iloc[index]['Genre']} ({df.iloc[index]['Platform']})",
                         expanded=False):
            for column, value in df.iloc[index].drop('ID').items():  # исключение столбца ID
                st.write(f"{column}: {value}")


# Отображение страницы получения рекомендаций
def recommendation_page():
    df_all = load_data("data/new_all_data.csv")
    similarity = vectorize_genre_to_cosine_mat(load_data("data/new_dataset.csv")['genre'])

    st.title('Получить рекомендации')
    st.write('Введите название видеоигры, чтобы получить рекомендации похожих игр по жанру:')

    selected_game = st.selectbox('Выберите видеоигру:', df_all['Title'].unique())

    search_button = st.button('Рекомендации')

    if search_button:
        if selected_game in df_all['Title'].values:
            recommendations = recommend(load_data("data/new_dataset.csv"), selected_game, similarity)
            if recommendations != "Game not found in the dataset":
                st.write('Рекомендации:')
                recommended_titles = set()
                for game_title, similarity_score in recommendations:
                    if game_title not in recommended_titles:
                        recommended_titles.add(game_title)
                        game_row = df_all[df_all['Title'] == game_title].iloc[0]
                        with st.expander(f"{game_row['Title']} - {game_row['Genre']} ({game_row['Platform']})", expanded=False):
                            for column, value in game_row.drop('ID').items():
                                st.write(f"{column}: {value}")
            else:
                st.write('Игра не найдена в наборе данных')
        else:
            st.write('Игра не найдена в наборе данных')


# Основная функция для выбора страницы приложения
def main():
    selected_page = st.sidebar.radio('Выберите страницу', ['Главная страница', 'Список видеоигр', 'Поиск видеоигры',
                                                           'Рекомендации по видеоигре', 'Разработчик'])

    if selected_page == 'Главная страница':
        main_page()
    elif selected_page == 'Список видеоигр':
        list_games_page()
    elif selected_page == 'Поиск видеоигры':
        video_game_page()
    elif selected_page == 'Рекомендации по видеоигре':
        recommendation_page()
    elif selected_page == 'Разработчик':
        about_page()


if __name__ == "__main__":
    main()