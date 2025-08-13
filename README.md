
# **Exploratory Data Analysis (EDA) Report: Chicago Crime Data**

## 📂 Download the PBIX File
[Click here to download the full Power BI Report](https://drive.google.com/file/d/1874bEhPUpiR4xgMc3Q6ZqmX_EnUvvMc1/view?usp=sharing)


## **1. Overview**

The dataset contains records of crimes committed in Chicago, including information such as location, time, type of crime, arrest details, and more. This report explores the dataset to uncover insights about crime patterns, hotspots, trends, and other aspects crucial for understanding and improving community safety.

---

## **2. Dataset Overview**

### **2.1 Key Statistics**
- **Total Records:** 549,960  
- **Number of Crime Types:** 27  
- **Number of Locations:** Multiple, based on latitude and longitude  
- **Time Range:** 2001 to 2024  


### **2.2 Key Features**
**Categorical Features:**
- **Primary Type, Description, Location Description, Season, Day_of_Week.**

**Numerical Features:**
- **Latitude, Longitude, Hour, Year, Cluster.**

**Temporal Features:**
- **Date, Updated On.**

**Missing Values**
- **The dataset has no missing values, ensuring data completeness for analysis.**

---

## **3. Insights and Graphs**

### **3.1 Temporal Analysis**

#### **3.1.1 Crime Trends Over Time**
**Insight:**  
Crime rates have shown a decrease from 2001 to 2024, with a noticeable peak in 2023 due to possible data anomalies or reporting shifts. The overall crime trend, however, is declining in recent years.

**Graph:**  
Insert a **line graph** showing the crime count trend by year, month, or day.

#### **3.1.2 Peak Crime Hours**
**Insight:**  
Most crimes occur between **6 PM and 10 PM**, likely due to increased activity in public spaces during the evening hours.

**Graph:**  
Insert a **heatmap** with time on the y-axis and days of the week on the x-axis.

---

### **3.2 Crime Type Analysis**

#### **3.2.1 Distribution of Crime Types**
**Insight:**  
The most common crimes are **Theft** (20.5%) and **Battery** (18.4%), together accounting for **38.9%** of total crimes.

**Graph:**  
Insert a **bar chart** or **tree map** to show the distribution of crime types.

#### **3.2.2 Severity Analysis**
**Insight:**  
Severe crimes like **Homicide** and **Sex Offense** are declining over the years but still account for a significant portion of total crimes in high-risk areas.

**Graph:**  
Insert a **stacked bar chart** showing severe vs. non-severe crimes.

---

### **3.3 Geospatial Analysis**

#### **3.3.1 Crime Hotspots**
**Insight:**  
Crime hotspots are observed in **downtown Chicago** and surrounding neighborhoods like **Near North Side** and **Austin**.

**Graph:**  
Insert a **heatmap** showing crime density using latitude and longitude.

#### **3.3.2 District/Ward Analysis**
**Insight:**  
District **1** (near downtown) has the highest crime rate, while District **19** has the lowest.

**Graph:**  
Insert a **choropleth map** comparing crime rates across districts.

---

### **3.4 Arrest Analysis**

#### **3.4.1 Arrest Rates**
**Insight:**  
The overall arrest rate has dropped significantly from **29.5% in 2001** to **12.0% in 2023**, with a noticeable decrease in arrests for **battery** and **theft**.

**Graph:**  
Insert a **donut chart** or **bar chart** for arrest rates by crime type.

#### **3.4.2 Arrest Efficiency by Location**
**Insight:**  
Areas like **Near North Side** have high arrest rates, while neighborhoods like **Chatham** have a low arrest rate, suggesting a potential gap in law enforcement resources.

**Graph:**  
Insert a **grouped bar chart** or **highlight table** showing arrest rates by location.

---

### **3.5 Seasonal and Weather Impact Analysis**

#### **3.5.1 Seasonal Trends**
**Insight:**  
Crime rates spike during **summer months**, especially in **June** and **July**, likely due to warmer weather and increased public activity.

**Graph:**  
Insert a **line graph** showing crime trends by season.

---

### **3.6 Domestic vs. Non-Domestic Crimes**

**Insight:**  
Domestic crimes account for **7.5%** of total crimes. **Battery** and **assault** are the most common domestic crimes.

**Graph:**  
Insert a **side-by-side bar chart** comparing domestic vs. non-domestic crimes.

---

### **3.7 Location Analysis**

#### **3.7.1 Common Crime Locations**
**Insight:**  
The most common crime locations are **street corners** and **residential areas**.

**Graph:**  
Insert a **horizontal bar chart** showing common locations and crime types.

#### **3.7.2 Repeat Crime Locations**
**Insight:**  
Certain locations, such as **the 74XX N Rogers Ave**, have repeated crimes, indicating a need for improved surveillance and preventive measures.

**Graph:**  
Insert a **clustered map** or **highlight table** for repeat crime locations.

---

### **3.8 Risk Assessment**

#### **3.8.1 Neighborhood Safety Scores**
**Insight:**  
Neighborhoods such as **Englewood** and **West Englewood** have low safety scores and need focused safety measures.

**Graph:**  
Insert a **conditional formatted map or table** with dynamic safety scores.

---

## **4. Key Findings**
- **Hotspots:** **Near North Side** and **Austin** are the most crime-prone areas.  
- **Peak Times:** Most crimes occur during **6 PM - 10 PM**, especially on **weekends**.  
- **Arrest Rates:** Arrest efficiency varies significantly across **crime types** and **locations**.  
- **Crime Types:** **Theft** and **Battery** are the most frequently occurring crimes.  
- **Seasonal Trends:** Crime peaks during **summer months** due to higher outdoor activity.  

---

## **5. Recommendations**
1. Increase patrols in high-crime areas such as **Near North Side** and **Austin**.  
2. Focus on improving arrest efficiency for **Battery** and **Theft** in areas with lower arrest rates like **Chatham**.  
3. Enhance safety measures in **low-safety neighborhoods** such as **Englewood** and **West Englewood**.  
4. Develop awareness campaigns for **Battery** and **Assault** in **high-incidence areas**.  

---

## **6. Future Scope**
- Implement predictive models to forecast future crime patterns based on historical data.  
- Analyze additional datasets (e.g., weather, economic conditions) to identify external factors influencing crime.  
- Create a mobile-friendly dashboard for real-time crime tracking and analysis.
