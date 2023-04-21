library(tidyverse)
mpg

ggplot(data=mpg, aes(x=cyl, y=hwy)) + geom_point()
ggplot(data=mpg, aes(x=cyl, y=hwy)) + geom_smooth()


fig9 <- ggplot(data = mpg, mapping = aes(x = displ, y = hwy, color=class)) +
  geom_point() +
  geom_smooth(aes(linetype = class), se = FALSE)
fig9 + labs(title= "Engine Size vs Fuel Efficiency", 
            subtitle= "Figure 9",
            x = "Engine Size (L)",
            y = "Fuel Efficiency (mpg)") + theme(legend.position = "top")


ggplot(mpg, aes(x=hwy)) +
  geom_density() + 
  geom_histogram(aes(y = ..density..),
                 bins = 25, color = "black",
                 fill = "purple", alpha = 0.2)+
  labs(x= 'Highway Fuel Consumption (MpG)', title = 'Histogram for Highway Fuel Consumption')

ggplot(mpg, aes(x=hwy)) +
  geom_histogram(aes(y = ..density..),
                 bins = 10, color = "black",
                 fill = "purple", alpha = 0.2)+ 
  stat_function(fun = dnorm, args = list(mean = mean(mpg$hwy), sd = sd(mpg$hwy)))+
  labs(x= 'Highway Fuel Consumption (MpG)', title = 'Histogram for Highway Fuel Consumption')

ggplot(mpg, aes(x=hwy)) +
  geom_histogram(aes(y = ..density..),
                 bins = 25, color = "black",
                 fill = "purple", alpha = 0.2) + 
  stat_function(fun = dnorm, args = list(mean = mean(mpg$hwy), sd = sd(mpg$hwy))) +
  labs(x= 'Highway Fuel Consumption (MpG)', title = 'Histogram for Highway Fuel Consumption')



df <- mpg %>%
  group_by(hwy) %>%
  summarise(mean = mean(displ))


data <- data.frame(
  murder = USArrests$Murder,
  state = tolower(rownames(USArrests)))
data %>% head(3)

ggplot(data, aes(fill = murder)) +
  geom_map(aes(map_id = state), map = map, color='black') +
  expand_limits(x = map$long, y = map$lat) +
  scale_fill_gradient(low = 'pink', high='darkred') + 
  labs(x= 'Longitude', y='Latitude', title='USA Murder Rate') +
  theme_bw()

