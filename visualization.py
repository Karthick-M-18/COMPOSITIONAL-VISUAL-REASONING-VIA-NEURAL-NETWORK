import json
import pandas
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#from scipy.misc import imread, imresize
from imageio import imread
from skimage.transform import resize as imresize
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from PIL import Image as PILImage
import glob
from IPython.display import Image as DispImage, display
import os
#if os.path.exists(delfname):
#    os.remove(delfname)

flatten = lambda ll: [e for l in ll for e in l]

parser = argparse.ArgumentParser()

# experiment settings
parser.add_argument("--tier",  default = "val", choices = ["train", "val", "test"], type = str)
parser.add_argument("--expName",  default = "experiment", type = str)

# plotting
parser.add_argument("--cmap", default = "custom", type = str) # "gnuplot2", "GreysT" 

parser.add_argument("--trans", help = "transpose question attention", action = "store_true")
parser.add_argument("--sa", action = "store_true")
parser.add_argument("--gate", action = "store_true")

# filtering
parser.add_argument("--instances", nargs = "*", type = int)
parser.add_argument("--maxNum", default = 0, type = int)

parser.add_argument("--filter", default = [], nargs = "*", choices = ["mod", "length", "field"])
parser.add_argument("--filterMod", action = "store_true")
parser.add_argument("--filterLength", type = int) # 19
parser.add_argument("--filterField", type = str)
parser.add_argument("--filterIn", action = "store_true")
parser.add_argument("--filterList", nargs = "*") # ["how many", "more"], numbers

args = parser.parse_args()

isRight = lambda instance: instance["answer"] == instance["prediction"]
isRightStr = lambda instance: "RIGHT" if isRight(instance) else "WRONG"

# files
# jsonFilename = "valHPredictions.json" if args.humans else "valPredictions.json"
imagesDir = "./CLEVR_v1/images/{tier}".format(
    tier = args.tier)

dataFile = "./preds/{expName}/{tier}Predictions-{expName}.json".format(
    tier = args.tier, 
    expName = args.expName)

inImgName = lambda index: "{dir}/CLEVR_{tier}_{index}.png".format(
    dir = imagesDir, 
    index = ("000000%d" % index)[-6:],
    tier = args.tier)

outImgAttName = lambda instance, j: "./preds/{expName}/{tier}{id}Img_{step}.png".format(
    expName = args.expName, 
    tier = args.tier, 
    id = instance["index"], 
    step = j + 1)

outTableAttName = lambda instance, name: "./preds/{expName}/{tier}{id}{tableName}_{right}{orientation}.png".format(
    expName = args.expName, 
    tier = args.tier, 
    id = instance["index"], 
    tableName = name, 
    right = isRightStr(instance), 
    orientation = "_t" if args.trans else "")

# plotting
imageDims = (14,14)
figureImageDims = (2,3)
figureTableDims = (5,4)
fontScale = 1

# set transparent mask for low attention areas  
# cdict = plt.get_cmap("gnuplot2")._segmentdata
cdict = {"red": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1)), 
    "green": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1)), 
    "blue": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1))}
cdict["alpha"] = ((0.0, 0.35, 0.35),
                  (1.0,0.65, 0.65))
cmap = ListedColormap(cdict["red"], name="custom")
plt.register_cmap(cmap=cmap,override_builtin=True)

fname = "./Karthick_Gif_Outputs/{}".format(args.expName)
#delfname = "./output.html"
if not os.path.exists(fname):
    os.makedirs(fname)
    print(f"The folder '{fname}' has been created.".format(fname))
    
def savePlot(fig, fileName):
    plt.savefig(fileName, dpi = 720)
    plt.close(fig) 
    del fig

def filter(instance):
    if "length" in args.filter: 
        if len(instance["question"].split(" ")) > args.filterLength:
             return True

    if "field" in args.filter:
        if args.filterIn:  
            if not (instance[args.filterField] in args.filterList):
                return True
        else:
            if not any((l in instance[args.filterField]) for l in args.filterList):
                return True            

    if "mod" in args.filter:
        if (not isRight(instance)) and args.filterMod:
            return True

        if isRight(instance) and (not args.filterMod):
            return True

    return False

def showImgAtt(img, instance, step, ax):
    dx, dy = 0.05, 0.05
    x = np.arange(-1.5, 1.5, dx)
    y = np.arange(-1.0, 1.0, dy)
    X, Y = np.meshgrid(x, y)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)

    ax.cla()

    img1 = ax.imshow(img, interpolation = "nearest", extent = extent)
    ax.imshow(np.array(instance["attentions"]["kb"][step]).reshape(imageDims), cmap = plt.get_cmap(args.cmap), 
        interpolation = "bicubic", extent = extent)

    ax.set_axis_off()
    plt.axis("off")

    ax.set_aspect("auto")


def showImgAtts(instance):
    img = imread(inImgName(instance["imageId"]))

    length = len(instance["attentions"]["kb"])
    
    # show images
    for j in range(length):
        fig, ax = plt.subplots()
        fig.set_figheight(figureImageDims[0])
        fig.set_figwidth(figureImageDims[1])              
        
        showImgAtt(img, instance, j, ax)
        
        plt.subplots_adjust(bottom = 0, top = 1, left = 0, right = 1)
        savePlot(fig, outImgAttName(instance, j))

def showTableAtt(instance, table, x, y, name):
    # if args.trans:
    #     figureTableDims = (len(y) / 2 + 4, len(x) + 2)
    # else:
    #     figureTableDims = (len(y) / 2, len(x) / 2)
    # xx = np.arange(0, len(x), 1)
    # yy = np.arange(0, len(y), 1)
    # extent2 = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
    
    fig2, bx = plt.subplots(1, 1) # figsize = figureTableDims
    bx.cla()

    sns.set(font_scale = fontScale)

    if args.trans:
        table = np.transpose(table)
        x, y = y, x
    
    tableMap = pandas.DataFrame(data = table, index = x, columns = y)
    
    bx = sns.heatmap(tableMap, cmap = "Purples", cbar = False, linewidths = .5, linecolor = "gray", square = True)
    
    # x ticks
    if args.trans:
        bx.xaxis.tick_top()
    locs, labels = plt.xticks()
    if args.trans:
        plt.setp(labels, rotation = 0)
    else:
        plt.setp(labels, rotation = 60)

    # y ticks
    locs, labels = plt.yticks()
    plt.setp(labels, rotation = 0)

    plt.savefig(outTableAttName(instance, name), dpi = 720)

def imageviewer(passednumber):
  from matplotlib.image import imread as myimread
  imNum = "000000" + passednumber
  imNum = imNum[-6:]
  img2 = myimread(fname="./CLEVR_v1/images/val/CLEVR_val_{}.png".format(imNum))
  width, height = 2.5, 2
  plt.clf()
  plt.imshow(img2,extent=[0, width, 0, height])
  plt.axis('off')
  plt.show()

def gifcreator(passedimg,passedid):
  imNum = passedimg
  base_image_path = './CLEVR_v1/images/val/CLEVR_val_{}.png'.format(imNum)
  base_image = PILImage.open(base_image_path)
  png_images = glob.glob('./preds/clevrExperiment/val{}Img_*.png'.format(passedid))
  png_images.sort()
  file_dic = {}
  for i in png_images:
    file_dic[i] = int(i.split("Img_")[-1].split(".png")[0])
  sorted_dict = dict(sorted(file_dic.items(), key=lambda item: item[1]))
  png_images = list(sorted_dict.keys())
  frames = []
  for png_image in png_images:
      overlay_image = PILImage.open(png_image)
      overlay_image = overlay_image.resize(base_image.size)
      mask = overlay_image.convert('L')
      overlay_image.putalpha(mask)
      composite_image = PILImage.alpha_composite(base_image.convert('RGBA'), overlay_image.convert('RGBA'))
      frames.append(composite_image)
  frames[0].save('./Karthick_Gif_Outputs/output_{}_{}.gif'.format(imNum,str(passedid)), format='GIF', append_images=frames[1:], save_all=True, duration=500, loop=0)

def gifViewer(passedimggif,passedidgif):
  imNum = passedimggif
  gif_path = "./Karthick_Gif_Outputs/output_{}_{}.gif".format(imNum,str(passedidgif))
  display(DispImage(filename=gif_path))

def main():
    with open(dataFile) as inFile:
        results = json.load(inFile)

    # print(args.exp)

    count = 0
    if args.instances is None:
        args.instances = range(len(results))

    for i in args.instances:        
        if filter(results[i]):
            continue

        if count > args.maxNum and args.maxNum > 0:
            break
        count += 1

        length = len(results[i]["attentions"]["kb"])
        showImgAtts(results[i])

        iterations = range(1, length + 1)
        questionList = results[i]["question"].split(" ")
        table = np.array(results[i]["attentions"]["question"])[:,:(len(questionList))]        
        showTableAtt(results[i], table, iterations, questionList, "text")

        if args.sa:
            iterations = range(length)
            sa = np.zeros((length, length))
            for i in range(length):
                for j in range(i+1):
                    sa[i][j] = results[i]["attentions"]["self"][i][j]
            
            showTableAtt(results[i], sa[i][j], iterations, iterations, "sa")                    
        line=[]
        imNum = "000000" + str(results[i]["imageId"])
        imNum = imNum[-6:]
        line.append(str(i))
        line.append("id:" + str(results[i]["index"]))
        line.append("img:"+ str(results[i]["imageId"]))
        line.append("Q:"+ str(results[i]["question"]))
        line.append("G:"+ str(results[i]["answer"]))
        line.append("P:"+ str(results[i]["prediction"]))
        line.append(isRightStr(results[i]))
        print(line[0])
        print(line[1])
        print(line[2])
        print(line[3])
        print(line[4])
        print(line[5])
        print(line[6])
        gifcreator(imNum,str(results[i]["index"]))
        gifViewer(imNum,str(results[i]["index"]))
        #line.append('<img src="./Karthick_Gif_Outputs/output_{}_{}.gif">'.format(str(imNum),str(results[i]["index"])))
        #line.append("-" * 100 + '<br>')
        #html_content = '<br>'.join(line)
        #with open("output.html", "a") as file:
        #  file.write(html_content)

        #imageviewer(str(results[i]["imageId"]))

        if args.gate:
            print(results[i]["attentions"]["gate"])

        print("________________________________________________________________________")

if __name__ == "__main__":
    main()
