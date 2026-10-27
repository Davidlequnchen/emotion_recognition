library(ARTool)
library(emmeans)
library(dplyr)
library(readxl)
dat <- read_excel('corpus information.xlsx')

dat$emotion <- factor(dat$emotion)
dat$subjectname <- factor(dat$subjectname)
dat$groupnumber <- factor(dat$groupnumber)
dat$Actress <- factor(dat$Actress)

# ART mixed model
model_recog <- art(
  Recognition_rate ~ emotion * subjectname +
    (1|groupnumber) + 
    (1|Actress),
  data = dat
)

anova_recog <- anova(model_recog)
print(anova_recog)

anova_recog

#partial eta squared
anova_tab <- anova_recog
F_values <- as.numeric(anova_tab$F)
df1 <- as.numeric(anova_tab$Df)
df2 <- as.numeric(anova_tab$Df.res)

eta_p2 <- (F_values * df1) /
  (F_values * df1 + df2)

anova_tab$eta_p2 <- eta_p2

print(anova_tab)

#posthoc compairsons
model_interaction <- artlm(model_recog, "emotion:subjectname")
library(emmeans)
emm_recog <- emmeans(model_interaction,
                     ~ subjectname | emotion)
pairs_recog <- pairs(emm_recog, adjust = "fdr")
summary(pairs_recog)

test(emm_recog, joint = TRUE)

#----------intensity---------------
library(lme4)
library(car)
library(lmerTest)

model_intensity <- lmer(
  Intensity ~ emotion * subjectname +
    (1|groupnumber) 
  # + (1|Actress)
  ,
  data = dat,
  REML = FALSE
)

anova_intensity <- anova(model_intensity, type = 3)
print(anova_intensity)

library(effectsize)
eta_results <- eta_squared(
  model_intensity,
  partial = TRUE
)

print(eta_results)

#planned comparisons
model_intensity3 <- lmer(
  Intensity ~ emotion * subjectname +
    (1|Actress),
  data = dat,
  REML = FALSE
)

emm <- emmeans(model_intensity3, ~ subjectname | emotion,pbkrtest.limit = 6675)

planned_contrasts <- list(
  "1st_vs_2nd" = c(0, 0.5, 0, 0.5, -0.5, -0.5),  
  "1st_vs_3rd" = c(-0.5, 0.5, -0.5, 0.5, 0, 0),  
  "2nd_vs_3rd" = c(-0.5, 0, -0.5, 0, 0.5, 0.5)   
)

contrast_results <- contrast(emm, planned_contrasts, adjust = "none")
summary(contrast_results)


#----------actor and rater's gender interaction-------

ACC <- read_excel('emotion_gender data.xlsx')


str(ACC)
ACC$actor <- factor(ACC$actor)
ACC$actor_gender <- factor(ACC$actor_gender)
ACC$rater_gender <- factor(ACC$rater_gender)

ACC$emotioncode

model2 <- art(correct ~ actor_gender*rater_gender + (1|group_number) + (1|actor), data = ACC)
anova_results <- anova(model2)
print(anova_results)
#happiness
ACC_happiness <- dplyr::filter(ACC,emotioncode == "happiness")
model2_happiness <- art(correct ~ actor_gender*rater_gender + (1|group_number) + (1|actor), data = ACC_happiness)
anova_results_happiness <- anova(model2_happiness)
print(anova_results_happiness)
#anger
ACC_anger <- dplyr::filter(ACC,emotioncode == "anger")
model2_anger <- art(correct ~ actor_gender*rater_gender + (1|group_number) + (1|actor), data = ACC_anger)
anova_results_anger <- anova(model2_anger)
print(anova_results_anger)
#disgust
ACC_disgust <- dplyr::filter(ACC, emotioncode == "disgust")
model_disgust <- art(correct ~ actor_gender * rater_gender,
                     data = ACC_disgust)
anova(model_disgust)
#fear
ACC_fear <- dplyr::filter(ACC, emotioncode == "fear")
model_fear <- art(correct ~ actor_gender * rater_gender,
                  data = ACC_fear)
anova(model_fear)
#sadness
ACC_sadness <- dplyr::filter(ACC, emotioncode == "sadness")
model_sadness <- art(correct ~ actor_gender * rater_gender,
                     data = ACC_sadness)
anova(model_sadness)
#surprise
ACC_surprise <- dplyr::filter(ACC, emotioncode == "surprise")
model_surprise <- art(correct ~ actor_gender * rater_gender,
                      data = ACC_surprise)
anova(model_surprise)
#neutral
ACC_neutral <- dplyr::filter(ACC, emotioncode == "neutral")
model_neutral <- art(correct ~ actor_gender * rater_gender,
                     data = ACC_neutral)
anova(model_neutral)
