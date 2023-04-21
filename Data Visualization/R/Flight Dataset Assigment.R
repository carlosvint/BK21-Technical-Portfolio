library(tidyverse)
library(nycflights13)

airlines
airports
planes 
weather

hist(na.omit(flights$dep_delay), main="Departure Delay Time for flights living New York in 2013",
     xlim=c(-50,400), xaxp=c(-50,400,9), xlab="Minutes (Negative times represent early departures)", ylab="Number of Flights", col="white")

hist(na.omit(flights$arr_delay), main="Arrival Delay Time for flights arriving New York in 2013",
     xlim=c(-50,400), xaxp=c(-50,400,9), xlab="Minutes (Negative times represent early departures)", ylab="Number of Flights", col="white")

delaydf <- flights %>%
  select(year:day, hour, tailnum, dep_delay, arr_delay) %>%
  mutate(manufacture_year = planes$year[match(tailnum, planes$tailnum)])


depdelaydf <- delaydf %>%
  na.omit() %>%
  filter(dep_delay>0)%>%
  group_by(manufacture_year) %>%
  summarise(ave_delay = mean(dep_delay), .groups = 'drop')

arrdelaydf <- delaydf %>%
  na.omit() %>%
  filter(arr_delay>0)%>%
  group_by(manufacture_year) %>%
  summarise(ave_delay = mean(arr_delay), .groups = 'drop')


df <- arrdelaydf
df$dep_ave_delay <- depdelaydf$ave_delay
#row.names(df) <- df$manufacture_year
matrixdf <- as.matrix(df[,2:3])
rownames(matrixdf) <- df$manufacture_year

colors = c("orange", "brown")


barplot(height=t(matrixdf), beside=TRUE, 
       main= 'Average Delay Time according to the plane Manufactured Year',
        xlab='Manufacture Year', ylab='Average Minutes Delayed', las=2, col = colors, 
       legend.text = c("Arrival delay", "Departure Delay"), args.legend=list(x=140, y=70))

delaydecade <- delaydf %>%
  mutate(decade = ifelse(manufacture_year<1960, "1950s", 
                         ifelse(manufacture_year<1970, "1960s",
                                ifelse(manufacture_year<1980, "1970s",
                                       ifelse(manufacture_year<1990, "1980s",
                                              ifelse(manufacture_year<2000, "1990s", 
                                                     ifelse(manufacture_year<2010, "2000s","2010s")))))))

decadedepdelaydf <- delaydecade %>%
  na.omit() %>%
  filter(dep_delay>0)%>%
  group_by(decade) %>%
  summarise(ave_delay = mean(dep_delay), .groups = 'drop')

decadearrdelaydf <- delaydecade %>%
  na.omit() %>%
  filter(arr_delay>0)%>%
  group_by(decade) %>%
  summarise(ave_delay = mean(arr_delay), .groups = 'drop')


decadedf <- decadearrdelaydf
decadedf$dep_ave_delay <- decadedepdelaydf$ave_delay
row.names(decadedf) <- decadedf$decade
decadematrixdf <- as.matrix(decadedf[,2:3])
rownames(decadematrixdf) <- decadedf$decade



barplot(height=t(decadematrixdf), beside=TRUE, legend=TRUE,
        main= 'Average Delay Time according to the plane Manufactured Decade',
        xlab='Manufactured Decade', ylab='Average Minutes Delayed', col = colors, 
        legend.text = c("Arrival delay", "Departure Delay"))


boxplot(na.omit(flights$dep_delay), na.omit(flights$arr_delay))


decadedepdelaydf <- delaydecade %>%
  na.omit() %>%
  group_by(decade) %>%
  summarise(count=n(), ave_delay = mean(dep_delay), .groups = 'drop')

decadearrdelaydf <- delaydecade %>%
  na.omit() %>%
  group_by(decade) %>%
  summarise(count=n(),ave_delay = mean(arr_delay), .groups = 'drop')











curiousdf <- flights %>%
  select(year:day, hour, tailnum, dep_delay, arr_delay, distance, carrier, air_time) %>%
  mutate(manufacture_year = planes$year[match(tailnum, planes$tailnum)])

dfa <- curiousdf %>%
  mutate(decade = ifelse(manufacture_year<1960, "1950s", 
                         ifelse(manufacture_year<1970, "1960s",
                                ifelse(manufacture_year<1980, "1970s",
                                       ifelse(manufacture_year<1990, "1980s",
                                              ifelse(manufacture_year<2000, "1990s", 
                                                     ifelse(manufacture_year<2010, "2000s","2010s")))))))
dfa %>%
  na.omit()%>%
  group_by(decade)%>%
  summarise(count=n(), avg_dis=mean(distance), avg_airtime=mean(air_time))
  

distance<- dfa %>%
  na.omit()%>%
  group_by(decade)%>%
  summarise(count=n(), avg_dis=mean(distance), avg_airtime=mean(air_time), 
            avg_arrdelay = mean(arr_delay), avg_depdelay=mean(dep_delay),
            max_arrdelay = max(arr_delay), max_depdelay=max(dep_delay))

christmas<-dfa %>%
  na.omit()%>%
  group_by(decade, month)%>%
  summarise(count=n(), avg_dis=mean(distance), avg_airtime=mean(air_time), 
            avg_arrdelay = mean(arr_delay), avg_depdelay=mean(dep_delay),
            max_arrdelay = max(arr_delay), max_depdelay=max(dep_delay))



arrdelaydf <- delaydf %>%
  na.omit() %>%
  filter(arr_delay>0)%>%
  group_by(manufacture_year) %>%
  summarise(avg_delay = mean(arr_delay), max_delay = max(arr_delay), counts=n())


d1 <- ggplot(arrdelaydf) + 
  geom_col(aes(x = manufacture_year, y = counts), size = 1, color = "orange", 
           fill = "white") +
  geom_line(aes(x = manufacture_year, y = 100*avg_delay), size = 1, color="brown") +
  labs(x='Year', y='Number of flights') + scale_y_continuous(sec.axis = sec_axis(~./100, name = "Average Delay"))

d1


  


arrival_delay <- df %>%
    na.omit() %>%
    filter(arr_delay>0) %>%
    mutate(DelayType = ifelse(arr_delay<15, "Didn't even notice",
                             ifelse(arr_delay<30, "Minor inconvenience",
                                ifelse(arr_delay<60, "100m Dash to the gate",
                                   ifelse(arr_delay<180, "Lost my next flight",
                                     ifelse(arr_delay<600, "Better get me a hotel", "The Terminal"))))))

dfa <- arrival_delay %>%
  mutate(decade = ifelse(manufacture_year<1960, "1950s", 
                         ifelse(manufacture_year<1970, "1960s",
                                ifelse(manufacture_year<1980, "1970s",
                                       ifelse(manufacture_year<1990, "1980s",
                                              ifelse(manufacture_year<2000, "1990s", 
                                                     ifelse(manufacture_year<2010, "2000s","2010s")))))))


finalarr<- arrival_delay %>%
     na.omit()%>%
     group_by(DelayType)%>%
     summarise(count=n(), age = mean(manufacture_year))

finalarr1<- dfa%>%
  na.omit()%>%
  group_by(DelayType, decade)%>%
  summarise(count=n(), median())