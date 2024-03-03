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
    st.write('Выберите критерии поиска:')
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

def vectorize_genre_to_cosine_mat(data):
    count_vect = CountVectorizer(encoding='utf-8')
    vectors = count_vect.fit_transform(data.values.astype('U'))
    cosine_sim_mat = cosine_similarity(vectors)
    return cosine_sim_mat

def recommend(data, title, similarity):
    if title in data['title'].values:
        game_index = data[data['title'] == title].index[0]
        distances = similarity[game_index]
        game_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        similar_games = sorted(list(zip(data['title'].iloc[[index for index, _ in game_list]], distances)), key=lambda x: x[1], reverse=True)[0:10]
        return similar_games
    else:
        return "Game not found in the dataset"

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
    df = load_data("data/new_dataset.csv")
    similarity = vectorize_genre_to_cosine_mat(df['genre'])  # переносим сюда инициализацию матрицы схожести

    st.title('Получить рекомендации')
    st.write('Введите название видеоигры, чтобы получить рекомендации похожих игр по жанру:')

    search_query = st.text_input('Введите название видеоигры:')
    search_button = st.button('Рекомендации')

    if search_button:
        recommendations = recommend(df, search_query, similarity)
        if recommendations != "Game not found in the dataset":
            st.write('Рекомендации:')
            recommended_titles = set()  # Создаем множество для отслеживания уже рекомендованных игр
            for game_title, similarity_score in recommendations:
                if game_title not in recommended_titles:  # Проверяем, не была ли игра уже рекомендована
                    recommended_titles.add(game_title)  # Добавляем игру в список рекомендованных
                    game_version = df.loc[df['title'] == game_title, 'platform'].values[0]  # Получаем версию игры
                    st.write(f"{game_title} ({game_version})")
        else:
            st.write('Игра не найдена в наборе данных')

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
