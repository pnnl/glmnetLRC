## Sorts a dataframe by one or more variables
## 
## Sorts a dataframe by one or more variables, ascending or descending
##  The order of the arguments \code{form} and \code{dat} is
##   interchangeable.  In the formula, use + for ascending, - for
##   decending. Sorting is left to right in the formula. 
##   For example, to sort the dataframe \code{Oats} with sorting variables \code{Block} and
##   \code{Variety}, useage is either of the following: 
## 
##   \code{sort.data.frame(~ Block - Variety, Oats)}  or
##   
##   \code{sort.data.frame(Oats, ~ - Variety + Block)}
##
## @export
## 
## @usage sort.data.frame(form, dat)
## sort.data.frame(dat, form)
## 
## @param form A formula with the variable names to use for sorting
## @param dat The dataframe to sort
## 
## @return  The sorted dataframe
## 
## @author Kevin Wright with ideas from Any Liaw and small edits by Landon Sego
## 
## @examples
## d <- data.frame(b=factor(c("Hi","Med","Hi","Low"),levels=c("Low","Med","Hi"), 
##                 ordered=TRUE),
##                 x=c("A","D","A","C"),y=c(8,3,9,9),z=c(1,1,1,2))
## 
## # Sort by descending z, descending b
## sort.data.frame(~-z-b,d)
## 
## # Sort by ascending x, ascending y, and ascending z
## sort.data.frame(~x+y+z,d)
## 
## # Sort by descending x, ascending y, ascending z
## sort.data.frame(~-x+y+z,d)
## 
## # Sort by ascending x, descending y, ascending z
## sort.data.frame(d,~x-y+z) 


sort.data.frame <- function(form,dat){ 
# Author: Kevin Wright 
# Some ideas from Andy Liaw 
# http://tolstoy.newcastle.edu.au/R/help/04/07/1076.html 


# Use + for ascending, - for decending. 
# Sorting is left to right in the formula 
   

# Useage is either of the following: 
# sort.data.frame(~Block-Variety,Oats) 
# sort.data.frame(Oats,~-Variety+Block) 
   

# If dat is the formula, then switch form and dat 
  if(inherits(dat,"formula")){ 
    f=dat 
    dat=form 
    form=f 
  } 
  if(form[[1]] != "~") 
    stop("Formula must be one-sided.") 

# Make the formula into character and remove spaces 
  formc <- as.character(form[2]) 
  formc <- gsub(" ","",formc) 
# If the first character is not + or -, add + 
  if(!is.element(substring(formc,1,1),c("+","-")))     formc <- paste("+",formc,sep="") 
# Extract the variables from the formula 
  vars <- unlist(strsplit(formc, "[\\+\\-]"))
  vars <- vars[vars!=""] # Remove spurious "" terms 

# Build a list of arguments to pass to "order" function 
  calllist <- list() 
  pos=1 # Position of + or - 
  for(i in 1:length(vars)){ 
    varsign <- substring(formc,pos,pos) 
    pos <- pos+1+nchar(vars[i]) 
    if(is.factor(dat[,vars[i]])){ 

      if(varsign=="-")
        calllist[[i]] <- -rank(dat[,vars[i]])
      else
        calllist[[i]] <- rank(dat[,vars[i]])

    } 
    else { 
      if(varsign=="-") {

        ####### Fix by Landon for character variables
        svar <- dat[,vars[i]]
        if (is.character(svar)) 
          calllist[[i]] <- -rank(svar)
        else
          calllist[[i]] <- -svar

        ########### Original Code, didn't work for character variables
#        calllist[[i]] <- - dat[,vars[i]]
        
        
      }
      else
        calllist[[i]] <- dat[,vars[i]]


    } 
  }

  return(dat[do.call("order", calllist), ])

} # sort.data.frame



#d = data.frame(b=factor(c("Hi","Med","Hi","Low"),levels=c("Low","Med","Hi"), 
#               ordered=TRUE),
#               x=c("A","D","A","C"),y=c(8,3,9,9),z=c(1,1,1,2))
#sort.data.frame(~-z-b,d)
#sort.data.frame(~x+y+z,d)


#sort.data.frame(~-x+y+z,d) 
#sort.data.frame(d,~x-y+z) 
