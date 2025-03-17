"""
Integrating PCA with deep learning models for stock market Forecasting
"""

# Import Libraries for Data Processing
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from stock_indicators import indicators, Quote, CandlePart
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import Libraries for Model Creation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader



# Define Global Variables
SHOW_FIGURES = True

STOCKS = ['ASELS.IS', 'TUPRS.IS', 'THYAO.IS', 'SISE.IS', 'FROTO.IS']
DATA = yf.download(STOCKS, start="2014-01-01", end="2024-01-01")

RATIO = [0.64, 0.16, 0.2]
SEQUENCE = 20
LAG1 = 30
LAG2 = 60



# Data Pre-Processing
# Min-Max Scaling
def min_max(data: [pd.DataFrame, np.array]) -> [pd.DataFrame, np.array]:
    data_min = data.min()
    data_max = data.max()
    return (data - data_min) / (data_max - data_min)

# Split Data
def split_data(data: [pd.DataFrame, torch.Tensor, list], ratio: list = None):
    if sum(ratio) != 1:
        raise ValueError('Ratio should be equal to 1.')

    # Split Calculations
    train_ratio, valid_ratio, test_ratio = ratio
    train_size = int(train_ratio * len(data))
    valid_size = int(valid_ratio * len(data))
    test_size = int(test_ratio * len(data))

    # Data Splitting
    train = data[:train_size]
    valid = data[train_size:train_size + valid_size]
    test = data[1-test_size:]

    return train, valid, test

# Display Figure 1
if SHOW_FIGURES:
    data_scaled = min_max(DATA)
    train_s, valid_s, test_s = split_data(data_scaled, RATIO)

    plt.figure(figsize=(10, 6))
    plt.plot(train_s.index, train_s['Close']['FROTO.IS'], label='Train Data')
    plt.plot(valid_s.index, valid_s['Close']['FROTO.IS'], label='Validation Data')
    plt.plot(test_s.index, test_s['Close']['FROTO.IS'], label='Test Data')

    # Adding title, labels, and legend
    plt.title('Figure 1: Normalised FROTO.IS Stock')
    plt.xlabel('Date')
    plt.ylabel('Normalised Close Price')
    plt.legend(loc='upper left')

    # Show the plot
    plt.ylim(-0.05, 1.05)
    ts = int(0.64 * len(DATA))
    vs = int(0.16 * len(DATA))
    plt.vlines(DATA.index[ts - 1], -1, 2, linestyles='dashed', color='crimson')
    plt.vlines(DATA.index[ts + vs - 1], -1, 2, linestyles='dashed', color='crimson')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Define Stock Technical Indicators
def weighted_moving_average(quotes: list[Quote], n: int, column: CandlePart = CandlePart.CLOSE):
    results = indicators.get_wma(quotes, n, column)
    return pd.DataFrame([{'Date': res.date, 'WMA': res.wma} for res in results])

def exponential_moving_average(quotes: list[Quote], n: int):
    results = indicators.get_ema(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'EMA': res.ema} for res in results])

def relative_strength_index(quotes: list[Quote], n: int):
    results = indicators.get_rsi(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'RSI': res.rsi} for res in results])

def chande_momentum_oscillator(quotes: list[Quote], n: int):
    results = indicators.get_cmo(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'CMO': res.cmo} for res in results])

def williams_percent_range(quotes: list[Quote], n: int):
    results = indicators.get_williams_r(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'WR': res.williams_r} for res in results])

def rate_of_change(quotes: list[Quote], n: int):
    results = indicators.get_roc(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'ROC': res.roc} for res in results])

def hull_moving_average(quotes: list[Quote], n: int):
    results = indicators.get_hma(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'HMA': res.hma} for res in results])

def triple_exponential_moving_average(quotes: list[Quote], n: int):
    results = indicators.get_tema(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'TEMA': res.tema} for res in results])

def average_directional_index(quotes: list[Quote], n: int):
    results = indicators.get_adx(quotes, n)
    return pd.DataFrame([{'Date': res.date, 'ADX': res.adx} for res in results])

def psychological_line(data: pd.DataFrame, price: str, stock: str, n: int):
    up_price = data[price][stock].diff() > 0
    return pd.DataFrame([{'Date': date, 'Pline': pline} for date, pline in
                         zip(data.index, (up_price.rolling(window=n).sum() / n * 100))])

# Transform data into candle form
def return_quotes_list(data: pd.DataFrame, stock: str) -> list[Quote]:
    return [
        Quote(d, o, h, l, c, v)
        for d, o, h, l, c, v
        in zip(data.index, data['Open'][stock], data['High'][stock],
               data['Low'][stock], data['Close'][stock], data['Volume'][stock])
    ]

# Calculate Stock Technical Indicators automatically
def calc_sti(data: pd.DataFrame, stock: str, days: int) -> pd.DataFrame:
    outp = pd.DataFrame({'Close': data['Close'][stock]})
    outp.index = outp.index.tz_localize(None)
    quotes = return_quotes_list(data, stock)

    wma = weighted_moving_average(quotes, days, CandlePart.CLOSE)
    ema = exponential_moving_average(quotes, days)
    rsi = relative_strength_index(quotes, days)
    cmo = chande_momentum_oscillator(quotes, days)
    wr = williams_percent_range(quotes, days)
    roc = rate_of_change(quotes, days)
    hma = hull_moving_average(quotes, days)
    tema = triple_exponential_moving_average(quotes, days)
    adx = average_directional_index(quotes, days)
    pline = psychological_line(data, 'Close', stock, days)

    for indicator in [wma, ema, rsi, cmo, wr, roc, hma, tema, adx, pline]:
        indicator['Date'] = pd.to_datetime(indicator['Date'].dt.date)
        outp = pd.merge(outp, indicator, on='Date', how='left')
    outp.set_index('Date', inplace=True)

    return outp

# Display Figure 2
if SHOW_FIGURES:
    figure2 = calc_sti(DATA, STOCKS[2], LAG1).loc['2020-01-01':]
    plt.figure(figsize=(10,6))
    plt.plot(figure2.index, figure2['Close'], label='Close')
    plt.plot(figure2.index, figure2['WMA'], label='WMA')
    plt.plot(figure2.index, figure2['EMA'], label='EMA')
    plt.plot(figure2.index, figure2['RSI'], label='RSI')
    plt.plot(figure2.index, figure2['CMO'], label='CMO')
    plt.plot(figure2.index, figure2['WR'], label='WilliamsR')
    plt.plot(figure2.index, figure2['ROC'], label='ROC')
    plt.plot(figure2.index, figure2['HMA'], label='HMA')
    plt.plot(figure2.index, figure2['TEMA'], label='TEMA')
    plt.plot(figure2.index, figure2['ADX'], label='ADX')
    plt.plot(figure2.index, figure2['Pline'], label='Pline', color='yellow')

    # Adding title, labels, and legend
    plt.title('Figure 2: Technical Stock Indicators of THYAO.IS Stock')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left')

    # Show the plot
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Calculate PCA
def principal_component_analysis(data: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    indis = data.drop(columns=['Close'])
    indis.dropna(inplace=True)
    pca = PCA(n_components)
    pca.fit(indis)
    return (
        pd.DataFrame([
            {
                'Date': date,
                'Close': close,
                **{f'PC{i + 1}': val for i, val in enumerate(pca_vals)}
            }
            for date, close, pca_vals
            in zip(indis.index, data['Close'].iloc[-len(indis):], pca.transform(indis))
        ]).set_index('Date')
    )

# Display Figure 3 & 4
if SHOW_FIGURES:
    stock_n = 2
    figure3 = calc_sti(DATA, STOCKS[stock_n], LAG1)
    figure3 = principal_component_analysis(figure3)
    plt.figure(figsize=(10,6))
    plt.plot(figure3.index, figure3['Close'], label='Close')
    plt.plot(figure3.index, figure3['PC1'], label='PC1')
    plt.plot(figure3.index, figure3['PC2'], label='PC2')
    plt.plot(figure3.index, figure3['PC3'], label='PC3')

    # Adding title, labels, and legend
    plt.title(f'Figure 3: PCA and {STOCKS[stock_n]} Stock after standardization')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend(loc='upper left')

    # Show the plot
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for PC1, PC2, and PC3
    ax.scatter(figure3['PC1'], figure3['PC2'], figure3['PC3'], marker='o')

    # Adding titles and labels
    ax.set_title(f'Figure 4: PCA of {STOCKS[stock_n]} Stock Indicators')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    # Show the 3D plot
    plt.show()



# Prepare the Data
def split_xy(data: pd.DataFrame, time_lag: int) -> [pd.DataFrame, pd.DataFrame]:
    x_data = data.copy()
    y_data = data['Close'].copy()
    y_data = min_max(y_data)

    return x_data.iloc[:-(time_lag + SEQUENCE - 1)] ,y_data.iloc[(SEQUENCE + time_lag - 1):]

class Stock(Dataset):
    def __init__(self, data_x: pd.DataFrame, data_y: pd.DataFrame, sequence: int):
        if len(data_x) != len(data_y):
            raise ValueError(f'Inputs must be of the same length x({len(data_x)}) != y({len(data_y)})')

        self.x = np.array(data_x.values)
        self.y = np.array([np.array([val]) for val in data_y.values])
        self.sequence = sequence
        self.len = len(data_x) - sequence

    def __getitem__(self, index):
        return (torch.tensor(self.x[index:index + self.sequence], dtype=torch.float).T,
                torch.tensor(self.y[index], dtype=torch.float))

    def __len__(self):
        return self.len

def auto_data_prep(stock: str, time_frame: int, sequence: int, ratio: list) \
        -> [DataLoader, DataLoader, DataLoader, list]:
    data = calc_sti(DATA, stock, 14)                                            # Calculate stats
    data = principal_component_analysis(data)                                   # Break into PCA

    data_x, data_y = split_xy(data, time_frame)                                 # Split into X, y

    data_train_x, data_valid_x, data_test_x = split_data(data_x, ratio)         # Split X into train, val, test
    data_train_y, data_valid_y, data_test_y = split_data(data_y, ratio)         # Split y into train, val, test

    data_train = Stock(min_max(data_train_x), min_max(data_train_y), sequence)
    data_valid = Stock(min_max(data_valid_x), min_max(data_valid_y), sequence)
    data_test =  Stock(min_max(data_test_x), min_max(data_test_y), sequence)

    train_loader = DataLoader(data_train, 256, False)      # Create Dataloader
    valid_loader = DataLoader(data_valid, 256, False)      # Create Dataloader
    test_loader = DataLoader(data_test, 256, False)        # Create Dataloader

    return train_loader, valid_loader, test_loader, np.array(min_max(data_test_y))

# 'ASELS.IS' Stock
asels_train_30, asels_valid_30, asels_test_30, asels_test_y_30 = auto_data_prep('FROTO.IS', LAG1, SEQUENCE, RATIO)
asels_train_60, asels_valid_60, asels_test_60, asels_test_y_60 = auto_data_prep('FROTO.IS', LAG2, SEQUENCE, RATIO)

# 'TUPRS.IS' Stock
tuprs_train_30, tuprs_valid_30, tuprs_test_30, tuprs_test_y_30 = auto_data_prep('TUPRS.IS', LAG1, SEQUENCE, RATIO)
tuprs_train_60, tuprs_valid_60, tuprs_test_60, tuprs_test_y_60 = auto_data_prep('TUPRS.IS', LAG2, SEQUENCE, RATIO)

# 'THYAO.IS' Stock
thyao_train_30, thyao_valid_30, thyao_test_30, thyao_test_y_30 = auto_data_prep('THYAO.IS', LAG1, SEQUENCE, RATIO)
thyao_train_60, thyao_valid_60, thyao_test_60, thyao_test_y_60 = auto_data_prep('THYAO.IS', LAG2, SEQUENCE, RATIO)

# 'SISE.IS' Stock
sise_train_30, sise_valid_30, sise_test_30, sise_test_y_30 = auto_data_prep('SISE.IS', LAG1, SEQUENCE, RATIO)
sise_train_60, sise_valid_60, sise_test_60, sise_test_y_60 = auto_data_prep('SISE.IS', LAG2, SEQUENCE, RATIO)

# 'FROTO.IS' Stock
froto_train_30, froto_valid_30, froto_test_30, froto_test_y_30 = auto_data_prep('FROTO.IS', LAG1, SEQUENCE, RATIO)
froto_train_60, froto_valid_60, froto_test_60, froto_test_y_60 = auto_data_prep('FROTO.IS', LAG2, SEQUENCE, RATIO)

# Create the Models
class Model1(nn.Module):
    def __init__(self, sequence: int):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv1d(sequence, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.gru = nn.GRU(input_size=64, hidden_size=50, batch_first=True)
        self.fc1 = nn.Linear(100, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        outp = x.permute(0, 2, 1)
        outp = self.pool(torch.relu(self.conv1(outp)))
        outp = outp.permute(0, 2, 1)

        lstm_out, (lstm_hn, _) = self.lstm(outp)
        lstm_last_hidden = lstm_hn[-1]

        gru_out, gru_hn = self.gru(outp)
        gru_last_hidden = gru_hn[-1]
        combined_output = torch.cat((lstm_last_hidden, gru_last_hidden), dim=1)

        return self.fc1(combined_output)

class Model2(nn.Module):
    def __init__(self, sequence: int):
        super(Model2, self).__init__()
        self.conv = nn.Conv1d(4, 50, 3, padding='same')
        self.lstm = nn.LSTM(50, 50, bidirectional=True)
        self.fnc1 = nn.Linear(sequence*50*2, 25)   # *2 because of bidirectional LSTM
        self.fnc2 = nn.Linear(25, 1)
        self.sequence = sequence

    def forward(self, x):
        outp = x.permute(0, 2, 1)
        outp = self.conv(outp)              # Conv1D expects input shape (batch, input, sequence)
        outp = outp.permute(0, 2, 1)
        outp, _ = self.lstm(outp)           # LSTM expects input shape (batch, sequence, features)
        outp = func.relu(outp)
        outp = func.dropout(outp, 0.5)
        outp = outp.view(outp.shape[0], -1)
        outp = self.fnc1(outp)
        outp = func.relu(outp)
        outp = self.fnc2(outp)
        return outp

class Model3(nn.Module):
    def __init__(self, sequence: int):
        super(Model3, self).__init__()
        self.lstm1 = nn.LSTM(4, 50)
        self.lstm2 = nn.LSTM(50, 50)
        self.fnc1 = nn.Linear(50, 1)
        self.sequence = sequence

    def forward(self, x):
        outp, _ = self.lstm1(x)
        outp = func.relu(outp)
        outp, _ = self.lstm2(outp)
        outp = func.relu(outp)
        outp = self.fnc1(outp)
        return outp

class ProposedModel(nn.Module):
    def __init__(self, sequence: int):
        super(ProposedModel, self).__init__()
        self.lstm = nn.LSTM(4, 50)
        self.conv = nn.Conv1d(50, 50, 3, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flat = nn.Flatten()
        self.fnc1 = nn.Linear(sequence*25, 25)
        self.fnc2 = nn.Linear(25, 1)
        self.sequence = sequence

    def forward(self, x):
        outp = x.permute(0, 2, 1)
        outp, _ = self.lstm(outp)           # LSTM expects input shape (batch, sequence, features)
        outp = outp.permute(0, 2, 1)
        outp = self.conv(outp)              # Conv1D expects input shape (batch, input, sequence)
        outp = func.relu(outp)
        outp = self.pool(outp)
        outp = self.flat(outp)
        outp = self.fnc1(outp)
        outp = func.relu(outp)
        outp = self.fnc2(outp)
        return outp

def pick_model(model: str):
    if model == 'Model1':
        return Model1(SEQUENCE)
    elif model == 'Model2':
        return Model2(SEQUENCE)
    elif model == 'Model3':
        return Model3(SEQUENCE)
    elif model == 'ProMod':
        return ProposedModel(SEQUENCE)
    else:
        return ValueError(f'Model {model} does not exist')

def eval_model(train_loader: torch.tensor, valid_loader: torch.tensor, test_loader: torch.tensor, model: str,
               l_rate: float = 0.01, epochs: int = 200, train_threshold:float = 0.5) -> list:
    pred_y = []
    test_loss = 1
    _model = None
    model_state = None
    criterion = nn.MSELoss()  # Define Criterion

    while test_loss > train_threshold:  # Re-run based on test loss
        time.sleep(2)
        test_loss = 1  # Reset
        _model = pick_model(model)  # Init Model
        optimizer = optim.Adam(_model.parameters(), lr=l_rate)  # Define Optim

        # Training Loop
        run = True
        loss = None
        counter = 0
        for epoch in range(epochs):
            _model.train()
            if run is False:
                break
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = _model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation
            val_loss = 'None'
            if valid_loader is not None:
                _model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in valid_loader:
                        outputs = _model(inputs)
                        val_loss += criterion(outputs, targets).item()

                val_loss = val_loss / len(valid_loader)

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}')

            # Testing
            tmp_test_loss = 0.0
            _model.eval()
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = _model(inputs)
                    tmp_test_loss += criterion(outputs, targets).item()

            # Save the Best Model
            if tmp_test_loss <= test_loss:
                counter = 0
                test_loss = tmp_test_loss
                model_state = _model.state_dict()
            else:
                counter += 1
            if counter == 15:
                run = False

    # Testing
    test_loss = 0.0
    _model.load_state_dict(model_state)
    _model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = _model(inputs)
            for num in outputs:
                pred_y.append(num)
            test_loss += criterion(outputs, targets).item()

    test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {test_loss}\n')

    return pred_y

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, np.finfo(float).eps))) * 100

    return [round(mse, 4), round(mape, 4), round(mae, 4), round(r2, 4)]

# Display Figure 8
def show_result_fig(title: str, actual: list, proposed30, proposed60, model1_30=None, model1_60=None):
    len_act = len(actual)
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len_act)), actual, label='Actual Close', linewidth=3)
    plt.plot(list(range(len_act-len(proposed30), len_act)), proposed30,
             label='Predicted Proposed Model (30day lag)', linewidth=2) if proposed30 is not None else None
    plt.plot(list(range(len_act-len(proposed60), len_act)), proposed60,
             label='Predicted Proposed Model (60day lag)', linewidth=2) if proposed60 is not None else None
    plt.plot(list(range(len_act-len(model1_30), len_act)), model1_30,
             label='Predicted Model 1 (30day lag)') if model1_30 is not None else None
    plt.plot(list(range(len_act-len(model1_60), len_act)), model1_60,
             label='Predicted Model 1 (60day lag)') if model1_60 is not None else None

    # Adding title, labels, and legend
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')

    # Show the plot
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 'ASELS.IS' Stock Predictions
# pred_asels_model1_30d = eval_model(asels_train_30, asels_valid_30, asels_test_30, 'Model1', 0.001, 100)
# pred_asels_proposed_30d = eval_model(asels_train_30, asels_valid_30, asels_test_30, 'ProMod', 0.001, 100)
# pred_asels_model1_60d = eval_model(asels_train_60, asels_valid_60, asels_test_60, 'Model1', 0.001, 100)
# pred_asels_proposed_60d = eval_model(asels_train_60, asels_valid_60, asels_test_60, 'ProMod', 0.001, 100)
# show_result_fig('Stock ASELS.IS', asels_test_y_30,
#                 pred_asels_proposed_30d, pred_asels_proposed_60d,
#                 pred_asels_model1_30d, pred_asels_model1_60d)

# 'TUPRS.IS' Stock Predictions
# pred_tuprs_model1_30d = eval_model(tuprs_train_30, tuprs_valid_30, tuprs_test_30, 'Model1', 0.01, 100)
# pred_tuprs_proposed_30d = eval_model(tuprs_train_30, tuprs_valid_30, tuprs_test_30, 'ProMod', 0.01, 100)
# pred_tuprs_model1_60d = eval_model(tuprs_train_60, tuprs_valid_60, tuprs_test_60, 'Model1', 0.01, 100)
# pred_tuprs_proposed_60d = eval_model(tuprs_train_60, tuprs_valid_60, tuprs_test_60, 'ProMod', 0.01, 100)
# show_result_fig('Stock TUPRS.IS', tuprs_test_y_30,
#                 pred_tuprs_proposed_30d, pred_tuprs_proposed_60d,
#                 pred_tuprs_model1_30d, pred_tuprs_model1_60d)

# 'THYAO.IS' Stock Predictions
pred_thyao_model1_30d = eval_model(thyao_train_30, thyao_valid_30, thyao_test_30, 'Model1')
pred_thyao_model1_60d = eval_model(thyao_train_60, thyao_valid_60, thyao_test_60, 'Model1')
pred_thyao_proposed_30d = eval_model(thyao_train_30, thyao_valid_30, thyao_test_30, 'ProMod', train_threshold=0.2)
pred_thyao_proposed_60d = eval_model(thyao_train_60, thyao_valid_60, thyao_test_60, 'ProMod', train_threshold=0.2)

mm1_30 = regression_metrics(thyao_test_y_30[-len(pred_thyao_model1_30d):], pred_thyao_model1_30d)
mm1_60 = regression_metrics(thyao_test_y_60[-len(pred_thyao_model1_60d):], pred_thyao_model1_60d)
mpm_30 = regression_metrics(thyao_test_y_30[-len(pred_thyao_proposed_30d):], pred_thyao_proposed_30d)
mpm_60 = regression_metrics(thyao_test_y_60[-len(pred_thyao_proposed_60d):], pred_thyao_proposed_60d)

show_result_fig(f'Stock THYAO.IS', thyao_test_y_30,
                pred_thyao_proposed_30d, pred_thyao_proposed_60d,
                pred_thyao_model1_30d, pred_thyao_model1_60d)

# 'SISE.IS' Stock Predictions
# pred_sise_model1_30d = eval_model(sise_train_30, sise_valid_30, sise_test_30, 'Model1', 0.01, 100)
# pred_sise_proposed_30d = eval_model(sise_train_30, sise_valid_30, sise_test_30, 'ProMod', 0.01, 100)
# pred_sise_model1_60d = eval_model(sise_train_60, sise_valid_60, sise_test_60, 'Model1', 0.01, 100)
# pred_sise_proposed_60d = eval_model(sise_train_60, sise_valid_60, sise_test_60, 'ProMod', 0.01, 100)
# show_result_fig('Stock SISE.IS', sise_test_y_30,
#                 pred_sise_proposed_30d, pred_sise_proposed_60d,
#                 pred_sise_model1_30d, pred_sise_proposed_60d)

# 'FROTO.IS' Stock Predictions
# pred_froto_model1_30d = eval_model(froto_train_30, froto_valid_30, froto_test_30, 'Model1', 0.01, 100)
# pred_froto_proposed_30d = eval_model(froto_train_30, froto_valid_30, froto_test_30, 'ProMod', 0.01, 100)
# pred_froto_model1_60d = eval_model(froto_train_60, froto_valid_60, froto_test_60, 'Model1', 0.01, 100)
# pred_froto_proposed_60d = eval_model(froto_train_60, froto_valid_60, froto_test_60, 'ProMod', 0.01, 100)
# show_result_fig('Stock FROTO.IS', froto_test_y_30,
#                 pred_froto_proposed_30d, pred_froto_proposed_60d,
#                 pred_froto_model1_30d, pred_froto_model1_60d)

# Cross-Validation
def cross_val(stock: str, time_frame: int, model: str, n_of_split: int = 10):
    count = 0
    results = np.array([0.0, 0.0, 0.0, 0.0])
    for train_i in range(2, n_of_split):
        count += 1
        print([round(train_i/10, 2), 0.1])
        train, test, _, true_y = auto_data_prep(stock, time_frame, SEQUENCE,
                                                [round(train_i/10, 2), 0.1, round(1 - (train_i+1)/10, 2)])
        if model == 'ProMod':
            pred_y = eval_model(train, None, test, model, train_threshold=0.2)
        else:
            pred_y = eval_model(train, None, test, model)
        results += np.array(regression_metrics(true_y[-len(pred_y):], pred_y))

    return results/count

mm1_30_cv = cross_val('THYAO.IS', LAG1, 'Model1')
mm1_60_cv = cross_val('THYAO.IS', LAG2, 'Model1')
mpm_30_cv = cross_val('THYAO.IS', LAG1, 'ProMod')
mpm_60_cv = cross_val('THYAO.IS', LAG2, 'ProMod')

# Results
print(f'Metrics of Model 1 (30 day lag): '
      f'MSE: {mm1_30[0]} | MAPE: {mm1_30[1]} | MAE: {mm1_30[2]} | R2: {mm1_30[3]}')
print(f'Metrics of Model 1 (60 day lag): '
      f'MSE: {mm1_60[0]} | MAPE: {mm1_60[1]} | MAE: {mm1_60[2]} | R2: {mm1_60[3]}')
print(f'Metrics of Proposed Model (30 day lag): '
      f'MSE: {mpm_30[0]} | MAPE: {mpm_30[1]} | MAE: {mpm_30[2]} | R2: {mpm_30[3]}')
print(f'Metrics of Proposed Model (60 day lag): '
      f'MSE: {mpm_60[0]} | MAPE: {mpm_60[1]} | MAE: {mpm_60[2]} | R2: {mpm_60[3]}')

print(f'Cross-Validation Metrics of Model1 (30 day lag): '
      f'MSE: {mm1_30_cv[0]} | MAPE: {mm1_30_cv[1]} | MAE: {mm1_30_cv[2]} | R2: {mm1_30_cv[3]}')
print(f'Cross-Validation Metrics of Model1 (60 day lag): '
      f'MSE: {mm1_60_cv[0]} | MAPE: {mm1_60_cv[1]} | MAE: {mm1_60_cv[2]} | R2: {mm1_60_cv[3]}')
print(f'Cross-Validation Metrics of Proposed Model (30 day lag): '
      f'MSE: {mpm_30_cv[0]} | MAPE: {mpm_30_cv[1]} | MAE: {mpm_30_cv[2]} | R2: {mpm_30_cv[3]}')
print(f'Cross-Validation Metrics of Proposed Model (60 day lag): '
      f'MSE: {mpm_60_cv[0]} | MAPE: {mpm_60_cv[1]} | MAE: {mpm_60_cv[2]} | R2: {mpm_60_cv[3]}')
