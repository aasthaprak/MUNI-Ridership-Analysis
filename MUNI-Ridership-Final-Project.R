## Data Loading, Cleaning and Preprocessing 

```{r, echo=TRUE, eval=FALSE}
# Load the dataset 
df <- read.csv("/Users/aasthaprakash/Downloads/RidershipTable.csv")

# Load libraries 
library(dplyr)
library(ggplot2)
library(knitr)
library(lmtest)
library(forecast)  

# Convert 'Average.Daily.Boardings' column to numeric
df$Average.Daily.Boardings <- as.numeric(gsub(",", "", df$Average.Daily.Boardings))

# Handle missing / invalid values
df$Average.Daily.Boardings[is.na(df$Average.Daily.Boardings)] <- 0

# Filter data for 'WEEKDAY' entries only 
df_weekday <- subset(df, Service.Day.of.the.Week == "WEEKDAY")

# Convert 'Month' to Date format for time series compatibility
df_weekday$Month <- as.Date(paste("01", df_weekday$Month), format = "%d %B %Y")

# Aggregate data by month and calculate the average of 'Average.Daily.Boardings'
monthly_ridership <- df_weekday %>%
  group_by(Month) %>%
  summarize(Average.Daily.Boardings = mean(Average.Daily.Boardings))

ggplot(monthly_ridership, aes(x = Month, y = Average.Daily.Boardings)) +
  geom_line() +
  labs(title = "Monthly Ridership Over Time", x = "Month", y = "Average Daily Boardings")
```

## Time series conversion 

```{r, echo=TRUE, eval=FALSE}
# Convert the data to a time series
ts_ridership <- ts(monthly_ridership$Average.Daily.Boardings, start = c(2016, 1), frequency = 12)
```

## Stationarity testing and differencing 

```{r, echo=TRUE, eval=FALSE}
# Stationarity check
ndiffs(ts_ridership)
diff_ts <- diff(ts_ridership) # apply first differencing

# Plot the differenced series
plot(diff_ts, main = "Differenced Time Series", ylab = "Change in Boardings", xlab = "Time")

# Set up a 1x2 layout for the plots (ACF and PACF side by side)
par(mfrow = c(1, 2))

# Plot the ACF
acf(diff_ts, main = "ACF of Differenced Series")

# Plot the PACF
pacf(diff_ts, main = "PACF of Differenced Series")
```

## SARIMA model

```{r, echo=TRUE, eval=FALSE}
# Fit SARIMA model
sarima_model <- arima(ts_ridership, order = c(0, 1, 1), seasonal = list(order = c(0, 1, 1), period = 12))

# Print summary of the model
summary(sarima_model)

# Residual diagnostics 
checkresiduals(sarima_model)

# Ljung-Box test on SARIMA residuals
Box.test(sarima_model$residuals, lag = 24, type = "Ljung-Box")

# Evaluate the accuracy of the forecast
accuracy(sarima_model)

# Forecast on the test data
forecast_s<- forecast(sarima_model, h = 12)

# Plotting the forecasts
plot(forecast_s, main = "Forecast with SARIMA", ylab = "Average Daily Boardings", xlab = "Time")

# Print summary of data and the upper and lower limits
summary(forecast_s)

# The forecast for the next 12 months
forecast_s

# Load necessary libraries
library(dplyr)
library(knitr)
library(zoo)  # for as.yearmon function

# Create a data frame of forecasted values
forecast_df <- data.frame(
  Month = as.yearmon(time(forecast_s$mean)),  # Convert time to Year-Month format using zoo::as.yearmon
  Forecast = forecast_s$mean,
  Lo_80 = forecast_s$lower[,1],  # 80% lower bound
  Hi_80 = forecast_s$upper[,1],  # 80% upper bound
  Lo_95 = forecast_s$lower[,2],  # 95% lower bound
  Hi_95 = forecast_s$upper[,2]   # 95% upper bound
)

# Round the values to the nearest whole number
forecast_df_rounded <- forecast_df %>%
  mutate(
    Forecast = round(Forecast),
    Lo_80 = round(Lo_80),
    Hi_80 = round(Hi_80),
    Lo_95 = round(Lo_95),
    Hi_95 = round(Hi_95),
    Month = format(Month, "%b %Y")  # Convert to 'Month Year' format
  )

# Print the table of forecasted values
knitr::kable(forecast_df_rounded, 
             col.names = c("Month", "Forecast", "80% CI Lower", "80% CI Upper", "95% CI Lower", "95% CI Upper"),
             caption = "Forecasted Ridership with Confidence Intervals (Rounded)",
             format = "markdown", 
             digits = 0)  # Ensures rounding to whole numbers
```


## Lagged Regression Model 

```{r, echo=TRUE, eval=FALSE}
# Ensure that the ts_ridership is a time series object
# If it is not already, make sure ts_ridership is converted to a time series object
ts_ridership <- ts(ts_ridership, frequency = 12, start = c(2016, 1))  # Example start date

# Create lagged variables for the ridership data
lagged_data <- data.frame(
  y = ts_ridership[13:length(ts_ridership)],  # Dependent variable (starting from time point 13)
  lag1 = ts_ridership[1:(length(ts_ridership)-12)],  # Lag 1 (previous month)
  lag2 = ts_ridership[2:(length(ts_ridership)-11)],  # Lag 2 (2 months ago)
  lag3 = ts_ridership[3:(length(ts_ridership)-10)],  # Lag 3 (3 months ago)
  lag4 = ts_ridership[4:(length(ts_ridership)-9)],   # Lag 4 (4 months ago)
  lag5 = ts_ridership[5:(length(ts_ridership)-8)]    # Lag 5 (5 months ago)
)

# Fit a linear model (lagged regression) with the lagged variables
lagged_model <- lm(y ~ lag1 + lag2 + lag3 + lag4 + lag5, data = lagged_data)

# Print the summary of the model to check the coefficients
summary(lagged_model)

# Prepare the forecast for the next 12 months
# We'll use the last 5 months of observed data to generate the forecast
last_observed <- tail(ts_ridership, 5)  # Last 5 months of data

# Prepare the input for the prediction (using lagged values from the last observed months)
new_data <- data.frame(
  lag1 = last_observed[5],  # Most recent month
  lag2 = last_observed[4],  # Second most recent
  lag3 = last_observed[3],  # Third most recent
  lag4 = last_observed[2],  # Fourth most recent
  lag5 = last_observed[1]   # Fifth most recent
)

# Make a forecast for the next 12 months
forecast_values <- numeric(12)  # Create a vector to store the forecast values
for (i in 1:12) {
  forecast_values[i] <- predict(lagged_model, newdata = new_data)  # Forecast for the next month
  # Update the input for the next forecast iteration (shift the lagged variables)
  new_data <- data.frame(
    lag1 = forecast_values[i], 
    lag2 = new_data$lag1, 
    lag3 = new_data$lag2, 
    lag4 = new_data$lag3, 
    lag5 = new_data$lag4
  )
}

# Plot the original data
plot(ts_ridership, type = "l", col = "blue", main = "SF Muni Ridership with Lagged Regression Forecast", 
     ylab = "Ridership", xlab = "Time", xlim = c(start(ts_ridership)[1], 2026))  # Adjust xlim to fit forecast range

# Create a proper time axis for the forecast
forecast_x <- seq(time(ts_ridership)[length(ts_ridership)] + 1/12, by = 1/12, length.out = 12)

# Add the forecasted values (starting from the 13th month)
lines(forecast_x, forecast_values, col = "red", lwd = 2)

# Add a legend
legend("topright", legend = c("Observed", "Forecasted"), col = c("blue", "red"), lty = 1)


# Create a data frame for the forecasted values
forecast_lagged_df <- data.frame(
  Month = as.yearmon(time(ts_ridership)[length(ts_ridership)] + (1:12)/12),  # Time for next 12 months
  Forecast = round(forecast_values)
)

# Print the table of forecasted values for the next 12 months
knitr::kable(forecast_lagged_df, 
             col.names = c("Month", "Forecast"), 
             caption = "Lagged Regression Forecast for Next 12 Months", 
             format = "markdown", 
             digits = 0)

# model diagnostics 
hist(residuals(lagged_model), main = "Residuals Histogram", xlab = "Residuals")

qqnorm(residuals(lagged_model))
qqline(residuals(lagged_model), col = "red")

plot(residuals(lagged_model), main = "Residuals of the Lagged Regression Model")
abline(h = 0, col = "red")
points(which(abs(residuals(lagged_model)) > 2 * sd(residuals(lagged_model))), residuals(lagged_model)[abs(residuals(lagged_model)) > 2 * sd(residuals(lagged_model))], col = "blue", pch = 19)
```


