import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.dates import date2num
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Nasdaq 100 companies stored in a list below called tickers
tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'COST', 'PEP', 'NFLX', 'AMD'
    , 'CSCO', 'INTC', 'TMUS', 'CMCSA', 'INTU', 'QCOM', 'AMGN', 'TXN', 'HON', 'AMAT', 'SBUX', 'ISRG', 'BKNG',
           'MDLZ', 'LRCX', 'ADP', 'GILD', 'ADI', 'VRTX', 'REGN', 'MU', 'SNPS', 'PANW', 'PDD', 'MELI', 'KLAC', 'CDNS',
           'CSX', 'MAR', 'PYPL', 'CHTR', 'ASML', 'ORLY', 'MNST', 'CTAS', 'ABNB', 'LULU', 'NXPI', 'WDAY', 'CPRT', 'MRVL',
           'PCAR', 'CRWD', 'KDP', 'MCHP', 'ROST', 'ODFL', 'DXCM', 'ADSK', 'KHC', 'PAYX', 'FTNT', 'AEP', 'SGEN', 'CEG',
           'IDXX', 'EXC', 'AZN', 'EA', 'CTSH', 'FAST', 'VRSK', 'CSGP', 'BKR', 'DDOG', 'BIIB', 'XEL', 'GFS',
           'TTD', 'ON', 'MRNA', 'ZS', 'TEAM', 'FANG', 'WBD', 'ANSS', 'DLTR', 'EBAY', 'SIRI', 'WBA', 'ALGN', 'ZM', 'ILMN'
    , 'ENPH', 'JD', 'LCID']

# Creating variables to be used to set dates to download data within a 1-year timeframe.
end_date = datetime(2023, 12,
                    26)  # Hardcoding the date as I am getting null errors from a specific stock after 26th Dec
start_date = end_date - timedelta(365)

# Empty dataframe which will be used to store the Adjusted close values for each Nasdaq 100 company that is stored in
# the tickers list.
adjClose_data = pd.DataFrame()

# Loop through each ticker within the tickers list and download 1 year adjusted close prices for that individual ticker,
# once obtained put that information in the new data frame that I created above
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adjClose_data[ticker] = data['Adj Close']

# Utilising Pandas to display all rows to show all stocks
pd.set_option('display.max_rows', None)

# Transposing the data to get the right number of rows and columns for the assessment requirements
transposed_adjClose_data = adjClose_data.T


def pca_reduction_and_kmeans_clustering():  # Task 2 PCA reduction and Clustering Task
    # I display the shape off the dataframe before performing PCA reduction to showcase how many columns and rows are
    # in the dataframe before PCA operation.
    st.subheader("Dimensional Reduction utilising PCA")
    st.write("I used the code snippet below to display the shape of the dataframe before performing PCA reduction "
             "operation to showcase how many columns and rows are in the dataframe before attempting PCA reduction "
             "operation.")
    st.code("""f"Before PCA reduction the shape of the data frame is{transposed_adjClose_data.shape}""""")
    st.markdown("Here is the output from the terminal running the code above: **Before PCA reduction the shape of the "
                f"data frame is{transposed_adjClose_data.shape}**")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transposed_adjClose_data)

    # Reducing the data
    pca = PCA(n_components=10)
    pca_reduced_data = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_

    st.write(
        "I then perform PCA reduction operation with the code snippet below to reduce the columns from 250 to 10 columns.")
    st.code("""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transposed_adjClose_data)

    pca = PCA(n_components=10)
    pca_reduced_data = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_
    """)
    # I display the results to the terminal to confirm that PCA reduction has been successful in reducing the columns
    # from 250 to 10 columns.
    st.write("After PCA reduction operation, I then use the code snippet below to confirm that the pca reduction "
             "operation was successful in reducing the number of columns from 250 to 10 columns. ")
    st.code("""f"After PCA reduction the shape of the data frame is{pca_reduced_data.shape}""")
    st.markdown("Here is the output from the terminal running the code above: **After PCA reduction the shape of the "
                f"data frame is{pca_reduced_data.shape}**")

    # Adding a column called Tickers which shows the names off the stocks within the pca_reduced_data_dataFrame
    pca_reduced_data_dataFrame = pd.DataFrame(data=pca_reduced_data,
                                              columns=[f"PC{i}" for i in range(1, pca_reduced_data.shape[1] + 1)])
    pca_reduced_data_dataFrame["Tickers"] = tickers

    # Terminal visualization of the reduced columns done by the PCA reduction operation.
    st.write("Below is a visualisation of the reduced columns within the terminal done by the PCA reduction operation."
             " Bear in mind I am only calling the first 5 stocks within this output by calling .head() method on my pca"
             " reduced dataframe")
    st.write(pca_reduced_data_dataFrame.head())

    # Data preprocessing to only use the 10 PCA reduced columns and ignore the ticker names
    pca_reduced_data_numeric_values = pca_reduced_data_dataFrame.iloc[:, 0:10]

    # Kmeans Clustering the stocks into 4 clusters
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(pca_reduced_data_numeric_values)

    # Creating a new dataframe to visualise which stock tickers represent in which cluster group number.
    kmeans_clustering_results_df = pd.DataFrame({'Ticker': tickers, 'Assigned Cluster': cluster_labels})

    # Cluster Lists for better visualisation in terminal and front end GUI solution.
    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []

    # Appending to specific list depending on the Assigned Cluster Number they have been assigned by KMeans Clustering.
    for index, row in kmeans_clustering_results_df.iterrows():
        if row['Assigned Cluster'] == 0:
            cluster0.append(row['Ticker'])
        elif row['Assigned Cluster'] == 1:
            cluster1.append(row['Ticker'])
        elif row['Assigned Cluster'] == 2:
            cluster2.append(row['Ticker'])
        elif row['Assigned Cluster'] == 3:
            cluster3.append(row['Ticker'])

    st.subheader("Clustering with KMeans")
    st.write("To group the stocks together into 4 separate groups I utilised the clustering algorithm KMeans to group"
             " the stocks together into 4 separate cluster groups. The code snippet below is how I utilised KMeans "
             "clustering algorithm to group the stocks into 4 separate cluster groups.")
    st.code("""
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(pca_reduced_data_numeric_values)
    """)

    st.write("Below are the cluster groups and what stocks belong to which cluster group")
    tab0, tab1, tab2, tab3 = st.tabs(["Cluster Group 0", "Cluster Group 1", "Cluster Group 2", "Cluster Group 3"])
    # Displaying Cluster Lists
    tab0.write(cluster0)
    tab1.write(cluster1)
    tab2.write(cluster2)
    tab3.write(cluster3)


# Empty dataframe to store the Adjusted Close values for my selected stocks.
# My selected stocks are [NVDA, AMD, BKNG, ORLY]
selected_stocks = pd.DataFrame()

for ticker in tickers:
    if ticker in ('NVDA', 'AMD', 'BKNG', 'ORLY'):
        selected_stock_data = yf.download(ticker, start=start_date, end=end_date)
        selected_stocks[ticker] = selected_stock_data['Adj Close']

# Obtain the correlation info for my selected stocks [NVDA, AMD, BKNG, ORLY].
selected_stocks_correlated = selected_stocks.corr()

# Obtain the correlation info for the whole dataset stocks to compare and correlate against my selected stocks
adjClose_data_correlated = adjClose_data.corr()


def top10_positive_negative_correlation():
    st.write("In this page I have displayed the top 10 positive and negative correlations for each of my selected stock"
             " against the whole nasdaq 100 stocks in a heatmap view. At the bottom of each heatmap there is an "
             "expander on the page click it to view the raw data of the positive/negative correlation for each of my "
             "selected stocks.")

    st.write("**My Selected Stocks are [NVDA, AMD, BKNG, ORLY]**")

    # Looping through each stock from my selected stocks and displaying the stock and their top 10 positive/negative
    # correlations from the entire dataset.
    for stock in selected_stocks:
        print(f"Top 10 Positive Correlations with {stock}:")
        st.subheader(f"Top 10 Positive Correlations with {stock}:")
        top10_positive_correlations_with_stock = adjClose_data_correlated[stock].sort_values(ascending=False).head(11)[
                                                 1:]
        print(top10_positive_correlations_with_stock)

        # Converting the top10_positive_correlations_with_stock to a DataFrame, so that I can plot the positive correlation
        # between my selected stocks that are positively correlated against stocks from the whole dataset.
        top10_positive_correlations_with_stock_df = pd.DataFrame(top10_positive_correlations_with_stock,
                                                                 columns=[stock])

        # Creating and displaying the heatmap off the positive correlations for my selected stocks against the stocks from
        # the whole dataset.
        plt.figure(figsize=(10, 8))
        sns.heatmap(top10_positive_correlations_with_stock_df, annot=True, cmap='coolwarm')
        plt.title(f"Top 10 Positive Correlations with {stock}")
        st.pyplot()
        with st.expander(f"**Click here to see raw data of the Top 10 Positive Correlations with {stock}**"):
            st.write(top10_positive_correlations_with_stock)

        print("_______________________________________________________________________________________________________")
        print(f"Top 10 Negative Correlations with {stock}:")
        st.subheader(f"Top 10 Negative Correlations with {stock}:")
        top10_negative_correlations_with_stock = adjClose_data_correlated[stock].sort_values().head(10)
        print(top10_negative_correlations_with_stock)

        # Converting the top10_negative_correlations_with_stock to a DataFrame, so that I can plot the negative correlation
        # between my selected stocks that are negatively correlated against stocks from the whole dataset.
        top10_negative_correlations_with_stock_df = pd.DataFrame(top10_negative_correlations_with_stock,
                                                                 columns=[stock])

        # Creating and displaying the heatmap off the negative correlations for my selected stocks against the stocks from
        # the whole dataset.
        plt.figure(figsize=(10, 8))
        sns.heatmap(top10_negative_correlations_with_stock_df, annot=True, cmap='coolwarm')
        st.pyplot()
        with st.expander(f"**Click here to see raw data of the Top 10 Negative Correlations with {stock}**"):
            st.write(top10_negative_correlations_with_stock)
        print("_______________________________________________________________________________________________________")


def correlation_matrix_between_my_selected_stocks():
    print("Correlation Info between my selected stocks")
    # Heatmap that I created to visualise the correlation matrix between my selected stocks [NVDA, AMD, BKNG, ORLY].
    plt.figure(figsize=(12, 10))
    print(selected_stocks_correlated)
    sns.heatmap(selected_stocks_correlated, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix Heatmap for my selected stocks")
    st.subheader("Correlation Matrix Heatmap for my selected stocks")
    st.write(
        "Below is a heatmap that displays the correlation matrix between my selected stocks [NVDA, AMD, BKNG, ORLY].")
    st.pyplot()


def time_series_plots_for_my_selected_stocks():
    # Creating and displaying a chart with a historical view of Adjusted Close prices for all my selected stocks.
    plt.figure(figsize=(12, 10))

    for stock in selected_stocks:
        plt.plot(selected_stocks.index, selected_stocks[stock], label=stock)

    plt.title("Time Series Plot of Adjusted Close Prices for my selected stocks")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Prices")
    plt.legend()
    st.subheader("Time Series Analysis of my selected stocks")
    st.write("Below is a Time Series Visual Analysis where I'm displaying and comparing Adjusted Close Prices for my "
             "selected stocks in the span of one year. Within this Time Series Analysis we can see that BKNG's Adjusted"
             " Close Prices is far higher in value than the other 3 selected stocks which are 'AMD', 'NVDA' and 'ORLY'")
    st.pyplot()


# Facebook Prophet Method prediction
def fb_prophet():
    # Resetting and clearing the data to be processed for the Facebook Prophet Method
    selected_stocks.reset_index(inplace=True)
    selected_stocks_Date = selected_stocks['Date']

    for stock in selected_stocks.iloc[:, 1:]:
        prophet = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )

        # Creating a new DataFrame with the required columns for Facebook Prophet Prediction
        new_prophetDF = pd.DataFrame({'ds': selected_stocks_Date, 'y': selected_stocks[stock]})
        prophet.fit(new_prophetDF)

        # Creating a DataFrame to be used for the prediction
        future = prophet.make_future_dataframe(periods=365)

        # Passing the future DataFrame to generate a forecast prediction for my selected stocks.
        forecast = prophet.predict(future)

        # Plot the predictions that were made by Facebook Prophet Market prediction
        fig = plot_plotly(prophet, forecast)
        fig.update_layout(xaxis_title="Dates", yaxis_title="Stock Prices",
                          title_text=f"Facebook Prophet Prediction for {stock}")
        st.plotly_chart(fig)
        with st.expander(f"**Click here to see FB Prophet's Predicted Adjusted Close Prices for: {stock}**"):
            st.write(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Adjusted Close Prices'}))


# LSTM Model Prediction.
def lstm():
    for stock in selected_stocks:
        # Normalizing the data using MinMax Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(selected_stocks[stock].values.reshape(-1, 1))

        # Splitting the test and train data of the scaled data.
        train_size = int(len(scaled_data) * 0.8)
        test_size = int(len(scaled_data)) - train_size
        train_data, test_data = scaled_data[0: train_size], scaled_data[train_size:len(scaled_data), :1]
        print(len(train_data), len(test_data))

        # Function created to create the dataset for LSTM prediction
        def create_dataset(data, time_steps):
            X, y = [], []
            for i in range(len(data) - time_steps):
                X.append(data[i:(i + time_steps)])
                y.append(data[i + time_steps])
            return np.array(X), np.array(y)

        time_steps = 10

        # Using the create dataset function to create the dataset that will be used for LSTM Prediction
        X_train, y_train = create_dataset(train_data, time_steps)
        X_test, y_test = create_dataset(test_data, time_steps)
        print(X_train[0], y_train[0])

        # Creating the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')

        # Training the model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1, shuffle=False)

        # Visualising the training and testing process validation during training the LSTM model that I have created above.
        fig_validation_loss = plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title(f"Validation loss for Stock:{stock}")
        plt.legend()
        st.pyplot(fig_validation_loss)

        y_prediction = model.predict(X_test)

        # Evaluation of the predicted results made by the LSTM model, the visualisation shows the historic data and then
        # presents the future prediction made by the LSTM model for each stock. Finally plot the predicted results
        fig_prediction = plt.figure(figsize=(10, 8))
        plt.plot(np.arange(0, len(y_train)), y_train, 'g', label="history")
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, marker='.', label="true")
        plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_prediction, 'r', label="prediction")
        plt.ylabel('Value')
        plt.xlabel('Time Step')
        plt.title(f"Plotting predictions with real historical data for Stock:{stock}")
        plt.legend()
        st.pyplot(fig_prediction)

        fig_single_prediction = plt.figure(figsize=(10, 8))
        plt.plot(y_test, marker='.', label="true")
        plt.plot(y_prediction, 'r', label="prediction")
        plt.ylabel('Value')
        plt.xlabel('Time Step')
        plt.title(f"Prediction vs Training data for Stock:{stock}")
        plt.legend()
        plt.show()
        st.pyplot(fig_single_prediction)


# ARIMA Model Prediction
def arima():
    for stock in selected_stocks:
        stock_prices = selected_stocks[stock]

        # Split the data into training data and testing data
        train_size = int(len(stock_prices) * 0.9)
        train_data = stock_prices[:train_size]
        test_data = stock_prices[train_size:]

        # Building the train and test data for the model
        history = [x for x in train_data]

        # Storing the stock prices predictions
        predictions = list()

        # Creating the Arima Model and fitting the Arima model ready for training.
        arima_model = ARIMA(history, order=(1, 1, 0))
        fitted_arima_model = arima_model.fit()
        forcasted_values = fitted_arima_model.forecast()[0]
        predictions.append(forcasted_values)
        history.append(test_data[0])

        # Rolling multiple forecasts
        for i in range(1, len(test_data)):
            # Prediction
            arima_model = ARIMA(history, order=(1, 1, 0))
            fitted_arima_model = arima_model.fit()
            forcasted_values = fitted_arima_model.forecast()[0]
            predictions.append(forcasted_values)
            observations = test_data[i]
            history.append(observations)

        # Plotting the results
        plt.figure(figsize=(12, 8))
        plt.plot(stock_prices, color='green', label='Train Stock Price')
        plt.plot(test_data.index, test_data, color='red', label='Real Stock Price')
        plt.plot(test_data.index, predictions, color='blue', label='Predicted Stock Price')
        plt.title(f'Stock Price Prediction for : {stock}')
        plt.legend()
        st.pyplot()

        with st.expander(f"**Click here to view the performance of the ARIMA Model prediction on Stock: {stock}**"):
            # Reporting Performance for the ARIMA model
            mse = mean_squared_error(test_data, predictions)
            st.write('MSE: ' + str(mse))
            mae = mean_absolute_error(test_data, predictions)
            st.write('MAE: ' + str(mae))
            rmse = math.sqrt(mean_squared_error(test_data, predictions))
            st.write('RMSE: ' + str(rmse))

        # Utilising my current Arima model to predict stock prices for the next 7 days
        future_arima_model = ARIMA(stock_prices, order=(1, 1, 0))
        fitted_future_arima_model = future_arima_model.fit()
        next7_forecasted_values = fitted_future_arima_model.forecast(steps=7)

        st.write("Below attempting to utilise the ARIMA model to give a 7 day future forecasted prediction of Adj Close"
                 f" Price for Stock: {stock}")
        # Plotting the forecasted predicted prices for each stock in the next 7 days
        plt.figure(figsize=(12, 8))
        plt.plot(stock_prices.index, stock_prices, label='Original Prices')
        forecast_dates = pd.date_range(start=stock_prices.index[-1], periods=8, freq='D')[1:]
        plt.plot(forecast_dates, next7_forecasted_values, color='red', label="Forecasted Prices")
        plt.title(f"7 Day forecasted prediction for: {stock}")
        plt.legend()
        plt.tight_layout()
        st.pyplot()


def user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock, future_days=365):
    # This function has been created to allow the user to select a stock within the tickers list to be able to get an
    # analysis forecasting prediction provided by fb_prophet for their selected stock depending on the choice of time.

    user_selected_stock_date = user_selected_stock.index
    user_selected_stock_prices = user_selected_stock[user_selected_stock.columns[0]]

    # Fetch today's latest stock data and then adding it to the user_selected_stock DataFrame
    latest_data = yf.download(user_selected_stock.columns[0], start=end_date, end=end_date)
    user_selected_stock = pd.concat([user_selected_stock, latest_data['Adj Close']])

    prophet = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )

    # Creating a new DataFrame with the required columns for Facebook Prophet Prediction
    newProphetDF = pd.DataFrame({'ds': user_selected_stock_date, 'y': user_selected_stock_prices})
    prophet.fit(newProphetDF)

    # Creating a new DataFrame to be will be used for the prediction, but for this prediction we will pass the period
    # as a variable so that we can utilise this function for forecast stock prices for the users inputted stock
    # against different time periods which are 7, 14 and 30 days.
    future = prophet.make_future_dataframe(periods=future_days)

    # Passing the future DataFrame with the future days variable that will be passed and changed depending on the
    # forecasting analysis on the users selected stock
    forecast = prophet.predict(future)

    # Plot the predictions that were made by Facebook Prophet Market prediction
    fig = plot_plotly(prophet, forecast)
    fig.update_layout(xaxis_title="Dates", yaxis_title="Stock Prices",
                      title_text=f"Facebook Prophet Prediction for {user_selected_stock.columns[0]}")
    st.subheader(f"Selected Stock: {user_selected_stock.columns[0]} Facebook Prophet Prediction Chart")
    st.plotly_chart(fig)

    # Accessing the last/latest 'Adj Close' price for the user-selected stock
    st.subheader(f"Selected Stock: {user_selected_stock.columns[0]} Latest Stock Information")
    last_stock_date_price = user_selected_stock.iloc[-1]
    last_stock_date = user_selected_stock.index[-1]
    st.write(f"**Latest Updated Stock Date: {last_stock_date}**")
    st.write(f"**Latest Stock Price: {last_stock_date_price.iloc[0]}**")
    print(f"Latest Stock Price: {last_stock_date_price.iloc[0]}")

    # Print the forecasted prices for the users selected stock depending on the future days that have been entered by
    # the user with the user_selected_stock_forecast_analysis function.
    forecast.rename(columns={'ds': 'Date', 'yhat': 'Adj Close Price'}, inplace=True)
    st.subheader(f"Forecasted Prices for Stock({user_selected_stock.columns[0]}) for the Next {future_days} Days:")
    st.write(forecast[['Date', 'Adj Close Price']].tail(future_days))

    # I am retrieving data for the different time variance depending on the future days variable which will be 7, 14, 30
    # days. I access the data within the forecast dataframe and then offset the data depending on the time variance from
    # the future_days variable, I then save this as a variable to be used for comparison for stock analysis.
    future_days_stock_price = \
    forecast.loc[forecast['Date'] == (last_stock_date + pd.DateOffset(days=future_days)), 'Adj Close Price'].iloc[0]

    # Display stock analysis to the user regarding if they should invest/buy into the stock or sell/not invest into the
    # stock depending on their time analysis.
    st.subheader("User Selected Stock Analysis")
    if last_stock_date_price.iloc[0] > future_days_stock_price:
        st.write("**I'd advise that you sell this stock or don't invest in this stock at all.**")
        st.write(f"**Latest stock price for ({user_selected_stock.columns[0]}) is: {last_stock_date_price.iloc[0]}.**")
        st.write(
            f"**Which is higher than the predicted stock price valued at: {future_days_stock_price} in {future_days} "
            "days time.**")
    elif last_stock_date_price.iloc[0] < future_days_stock_price:
        st.write("**I'd advise you to invest in this stock.**")
        st.write(f"**Latest stock price for ({user_selected_stock.columns[0]}) is: {last_stock_date_price.iloc[0]}.**")
        st.write(
            f"**Which is lower than the predicted stock price valued at: {future_days_stock_price} in {future_days}**"
            f" **days time**")


def user_selected_stock_forecast_analysis():
    stock_user_selection = st.text_input("Enter the stock into the text field that you would like to analyse (e.g AAPL)"
                                         " 👇").upper()
    user_selected_stock_DF = pd.DataFrame()
    user_selected_stock_data = yf.download(stock_user_selection, start=start_date, end=end_date)
    user_selected_stock_DF[stock_user_selection] = user_selected_stock_data['Adj Close']
    user_input = st.selectbox("Select the analysis duration:", [7, 14, 30])
    if st.button("Analyse Stock"):
        if stock_user_selection in tickers:
            if user_input == 7:
                st.write(f"Analysing and forecasting stock prices for {stock_user_selection} for the next 7 days")
                user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock_DF, future_days=7)
            elif user_input == 14:
                st.write(f"Analysing and forecasting stock prices for {stock_user_selection} for the next 14 days")
                user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock_DF, future_days=14)
            elif user_input == 30:
                st.write(f"Analysing and forecasting stock prices for {stock_user_selection} for the next 30 days")
                user_selected_stock_forecast_analysis_with_fbProphet(user_selected_stock_DF, future_days=30)
        else:
            st.write("Unable to find Stock information for that inputted Stock Code")


def linear_regression():
    for stock in selected_stocks:
        # Accessing the Adj Close prices for each of my selected stocks
        stock_price = selected_stocks[stock]

        # Split data into train and test set: 80% / 20%
        train, test = train_test_split(stock_price.to_frame(), test_size=0.20)

        # Reshape x and y train data for linear regression
        X_train = date2num(train.index).astype(float).reshape(-1, 1)
        y_train = train[stock]

        # Create Linear Regression Model and then fit the data to the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Visualize the Linear Regression predictions against the actual price data
        linear_regression_fig = plt.figure(figsize=(10, 6))
        plt.title(f"Linear Regression Prediction for Stock: {stock}")
        plt.scatter(X_train, y_train, edgecolors='w', label='Actual Price')
        plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xlabel('Date')
        plt.ylabel('Stock Adj Close Price')
        plt.legend()
        st.pyplot(linear_regression_fig)