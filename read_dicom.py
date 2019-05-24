
# coding: utf-8



import hashlib, os, threading
import numpy as np 
import math
import scipy.ndimage
from skimage import measure
from skimage import morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import axes3d
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, plot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from matplotlib import pyplot, cm
import matplotlib.pyplot as plt
import glob
import random
from dicompylercore import  dvh, dvhcalc
from dicompylercore.dicomparser import DicomParser 
from six import PY2, iterkeys, string_types, BytesIO
#from shapely.geometry import Polygon
from scipy import linalg
try:
    from pydicom.dicomio import read_file
    from pydicom.dataset import Dataset
except ImportError:
    from dicom import read_file
    from dicom.dataset import Dataset 




class dicommethodcore:
    def searchpath(path):
        #檢查目錄是否存在
        if os.path.isdir(path):
            files = []
            #os.walk 遞迴印出所有目錄跟檔名，原本就分三個層次(root,dirs,filenames)
            for root, dirs, filenames in os.walk(path):
                #做出file的路徑名
                files += map(lambda f:os.path.join(root, f), filenames)
        else:
            #!其實不確定這裏要幹嘛
            files =[path]
        return files
    def getdicomdatainfo(files):
        for n in range(len(files)):
            if (n == 0):
                patients = {}
                ptdata={}
                h_record={}
            if (os.path.isfile(files[n])):
                try:
                     dp = DicomParser(files[n])
                except (AttributeError, EOFError, IOError, KeyError):
                    pass
                    print("%s is not a valid DICOM file.", files[n])
                else:
                    #取病人資料:名字 ID 性別 生日
                    patientGe = dp.GetDemographics()
                    h1=hashlib.sha1((dp.ds.PatientID).encode('utf-8')).hexdigest()
                    h2=dp.ds.StudyDate
                    if not h1 in patients:
                        h_record[h1] ={}
                        patients[h1] = {}
                        ptdata[h1]={}
                    if not h2 in patients[h1]:
                        h_record[h1][h2]=[]
                        patients[h1][h2]={}
                        ptdata[h1][h2]={}
                        patients[h1][h2]['patientinfo']={}
                        patients[h1][h2]['patientinfo']['demographics'] = patientGe
                        if not 'studies' in patients[h1]:
                            patients[h1][h2]['patientinfo']['studies'] = {}
                            patients[h1][h2]['patientinfo']['series'] = {}
                    if not (dp.GetSOPClassUID() == 'rtdose'):
                        stinfo = dp.GetStudyInfo()
                        if not stinfo['id'] in patients[h1][h2]['patientinfo']['studies']:
                            patients[h1][h2]['patientinfo']['studies'][stinfo['id']] = stinfo

                    #if:是跑影像， elif是跑dose等其他檔案
                    if (('ImageOrientationPatient' in dp.ds) and not (dp.GetSOPClassUID() == 'rtdose')): 
                        seinfo = dp.GetSeriesInfo()
                        seinfo['numimages'] = 0
                        seinfo['modality'] = dp.ds.SOPClassUID.name
                        if not seinfo['id'] in patients[h1][h2]['patientinfo']['series']:
                            patients[h1][h2]['patientinfo']['series'][seinfo['id']] = seinfo
                            #h_record[h1][h2]={}
                            h_record[h1][h2] += [seinfo['id']]
                        if not 'images' in patients[h1][h2]:
                            patients[h1][h2]['images'] = {}
                            #ptdata[h1][h2]={}
                            ptdata[h1][h2]['images']={}
                        image = {}
                        image['id'] = dp.GetSOPInstanceUID()
                        image['filename'] = files[n]
                        image['series'] = seinfo['id']
                        image['referenceframe'] = dp.GetFrameOfReferenceUID()
                        patients[h1][h2]['patientinfo']['series'][seinfo['id']]['numimages'] = patients[h1][h2]['patientinfo']['series'][seinfo['id']]['numimages'] + 1
                        patients[h1][h2]['images'][image['id']] = image
                        if not seinfo['id'] in ptdata[h1][h2]['images']:
                            ptdata[h1][h2]['images'][seinfo['id']]=[]   
                        #Chris為了排順序用
                        ptdata[h1][h2]['images'][seinfo['id']].append(dp.ds)
                        # Create each RT Structure Set

                    elif dp.ds.Modality in ['RTSTRUCT']:
                    #dp.ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3')
                        if not 'structures' in patients[h1][h2]:
                            patients[h1][h2]['structures'] = {}
                            ptdata[h1][h2]['structures'] = {}
                        structure = dp.GetStructureInfo()
                        structure['id'] = dp.GetSOPInstanceUID()
                        structure['filename'] = files[n]
                        ser=dp.GetReferencedSeries()
                        structure['series'] = ser
                        structure['referenceframe'] = dp.GetFrameOfReferenceUID()
                        patients[h1][h2]['structures'][structure['id']] = structure
                        if not ser in ptdata[h1][h2]['structures']:
                            ptdata[h1][h2]['structures'][ser]=[]
                        ptdata[h1][h2]['structures'][ser].append(dp.ds)
                    elif dp.ds.Modality in ['RTPLAN']:
                        if not 'plans' in patients[h1][h2]:
                            patients[h1][h2]['plans'] = {}
                            ptdata[h1][h2]['plans']=[] 
                        plan = dp.GetPlan()
                        plan['id'] = dp.GetSOPInstanceUID()
                        plan['filename'] = files[n]
                        plan['series'] = dp.ds.SeriesInstanceUID
                        plan['referenceframe'] = dp.GetFrameOfReferenceUID()
                        plan['beams'] = dp.GetReferencedBeamsInFraction()
                        plan['rtss'] = dp.GetReferencedStructureSet()
                        patients[h1][h2]['plans'][plan['id']] = plan
                        ptdata[h1][h2]['plans'].append(dp.ds)
                    elif dp.ds.Modality in ['RTDOSE']:
                        if not 'doses' in patients[h1][h2]:
                            patients[h1][h2]['doses'] = {}
                            ptdata[h1][h2]['doses']=[]
                        dose = {}
                        dose['id'] = dp.GetSOPInstanceUID()
                        dose['filename'] = files[n]
                        dose['referenceframe'] = dp.GetFrameOfReferenceUID()
                        dose['hasdvh'] = dp.HasDVHs()
                        dose['hasgrid'] = "PixelData" in dp.ds
                        dose['summationtype'] = dp.ds.DoseSummationType
                        dose['beam'] = dp.GetReferencedBeamNumber()
                        dose['rtss'] = dp.GetReferencedStructureSet()
                        rtplan=dp.GetReferencedRTPlan()
                        dose['rtplan'] = rtplan
                        patients[h1][h2]['doses'][dose['id']] = dose
                        ptdata[h1][h2]['doses'].append(dp.ds)
                    else:
                        print("%s is a %s file and is not " +  "currently supported.",
                              files[n], dp.ds.SOPClassUID.name)    
        return patients, h_record,ptdata
    def resortimage(patients,s1,s2,si):
        if 'images' in patients[s1][s2]:
            sortedimages = []
            unsortednums = []
            sortednums = []
            images = patients[s1][s2]['images'][si]
            sort = 'IPP'
                # Determine if all images in the series are parallel
                # by testing for differences in ImageOrientationPatient
            parallel = True
            for i, item in enumerate(images):
                if (i > 0):
                    iop0 = np.array(item.ImageOrientationPatient)
                    iop1 = np.array(images[i-1].ImageOrientationPatient)
                    if (np.any(np.array(np.round(iop0 - iop1),
                    dtype=np.int32))):
                        parallel = False
                        break
                        # Also test ImagePositionPatient, as some series
                        # use the same patient position for every slice
                    ipp0 = np.array(item.ImagePositionPatient)
                    ipp1 = np.array(images[i-1].ImagePositionPatient)
                    if not (np.any(np.array(np.round(ipp0 - ipp1),
                    dtype=np.int32))):
                        parallel = False
                        break
            # If the images are parallel, sort by ImagePositionPatient
            if parallel:
                sort = 'IPP'
            else:
                # Otherwise sort by Instance Number
                if not (images[0].InstanceNumber == images[1].InstanceNumber):
                    sort = 'InstanceNumber'
                # Otherwise sort by Acquisition Number
                elif not (images[0].AcquisitionNumber == images[1].AcquisitionNumber):
                    sort = 'AcquisitionNumber'

                # Add the sort descriptor to a list to be sorted
            for i, image in enumerate(images):
                if (sort == 'IPP'):
                    unsortednums.append(image.ImagePositionPatient[2])
                else:
                    unsortednums.append(image.data_element(sort).value)

                    # Sort image numbers in descending order for head first patients
            if ('hf' in image.PatientPosition.lower()) and (sort == 'IPP'):
                sortednums = sorted(unsortednums, reverse=True)
                # Otherwise sort image numbers in ascending order
            else:
                sortednums = sorted(unsortednums)

                # Add the images to the array based on the sorted order
            for k, slice in enumerate(sortednums):
                for i, image in enumerate(images):
                    if (sort == 'IPP'):
                        if (slice == image.ImagePositionPatient[2]):
                            sortedimages.append(image)
                    elif (slice == image.data_element(sort).value):
                        sortedimages.append(image)

                # Save the images back to the patient dictionary
            
            return sortedimages
    def structure_method(structuredata):
        structures = {}
        for i in range(0,len(structuredata)):
            structures[i]={}
            strd=structuredata[i]        
            """Returns a dictionary of structures (ROIs)."""
            # Determine whether this is RT Structure Set file
            if not (strd.Modality == 'RTSTRUCT'):
                return structures

            # Locate the name and number of each ROI
            if 'StructureSetROISequence' in strd:
                for item in strd.StructureSetROISequence:
                    data = {}
                    number = int(item.ROINumber)
                    data['id'] = number
                    data['name'] = item.ROIName
                    print("Found ROI #%s: %s", str(number), data['name'])
                    structures[i][number] = data

            # Determine the type of each structure (PTV, organ, external, etc)
            if 'RTROIObservationsSequence' in strd:
                for item in strd.RTROIObservationsSequence:
                    number = item.ReferencedROINumber
                    if number in structures:
                        structures[i][number]['type'] = item.RTROIInterpretedType

            # The coordinate data of each ROI is stored within ROIContourSequence
            if 'ROIContourSequence' in strd:
                for roi in strd.ROIContourSequence:
                    number = roi.ReferencedROINumber

                    # Generate a random color for the current ROI
                    structures[i][number]['color'] = np.array((
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)), dtype=int)
                    # Get the RGB color triplet for the current ROI if it exists
                    if 'ROIDisplayColor' in roi:
                        # Make sure the color is not none
                        if not (roi.ROIDisplayColor is None):
                            color = roi.ROIDisplayColor
                        # Otherwise decode values separated by forward slashes
                        else:
                            value = roi[0x3006, 0x002a].repval
                            color = value.strip("'").split("/")
                        # Try to convert the detected value to a color triplet
                        try:
                            structures[i][number]['color'] =np.array(color, dtype=int)
                        # Otherwise fail and fallback on the random color
                        except:
                            print(
                                "Unable to decode display color for ROI #%s",
                                str(number))

                    # Determine whether the ROI has any contours present
                    if 'ContourSequence' in roi:
                        structures[i][number]['empty'] = False
                    else:
                        structures[i][number]['empty'] = True

            return structures
    def GetStructureCoordinates(self, roi_number):
        """Get the list of coordinates for each plane of the structure."""
        datarange=[]

        planes = {}
        # The coordinate data of each ROI is stored within ROIContourSequence
        #?可是沒有耶
        if 'ROIContourSequence' in self:
            for roi in self.ROIContourSequence:
                if (roi.ReferencedROINumber == int(roi_number)):
                    if 'ContourSequence' in roi:
                        # Locate the contour sequence for each referenced ROI
                        for c in roi.ContourSequence:
                            # For each plane, initialize a new plane dict
                            plane = {}
                            n=3

                            # Determine all the plane properties
                            plane['type'] = c.ContourGeometricType
                            plane['num_points'] = int(c.NumberOfContourPoints)
                            plane['data'] = [c.ContourData[i:i+n] for i in range(0, len(c.ContourData), n)]
                            '''plane['data'] =                                     self.GetContourPoints(c.ContourData)'''
                            """Parses an array of xyz points & returns a array of point dicts."""
                            '''def GetContourPoints(self, array):
                             n = 3
                             return [array[i:i+n] for i in range(0, len(array), n)]'''
                            datara=list(zip(*plane['data']))
                            if not datarange:
                                datarange=([min(datara[0]),min(datara[1]),max(datara[0]),max(datara[1])])
                            else:
                                if ((datarange[0]>(min(datara[0])))):
                                    datarange[0]=min(datara[0])
                                if ((datarange[2]<(max(datara[0])))):
                                    datarange[2]=max(datara[0])
                                if ((datarange[1]>(min(datara[1])))):
                                    datarange[1]=min(datara[1])
                                if ((datarange[-1]<(max(datara[1])))):
                                    datarange[-1]=max(datara[1])

                            # Each plane which coincides with an image slice
                            # will have a unique ID
                            if 'ContourImageSequence' in c:
                                # Add each plane to the planes dict
                                # of the current ROI
                                z = str(round(plane['data'][0][2], 2)) + '0'
                                if z not in planes:
                                    planes[z] = []
                                planes[z].append(plane)

        datarange=([(datarange[0],datarange[1]),(datarange[2],datarange[3])])
        return planes,datarange
    def CalculatePlaneThickness(self, planesDict):
        """Calculates the plane thickness for each structure."""
        planes = []
        # Iterate over each plane in the structure
        for z in iterkeys(planesDict):
            planes.append(float(z))
        planes.sort()

        # Determine the thickness
        thickness = 10000
        for n in range(0, len(planes)):
            if (n > 0):
                newThickness = planes[n] - planes[n-1]
                if (newThickness < thickness):
                    thickness = newThickness

            # If the thickness was not detected, set it to 0
        if (thickness == 10000):
            thickness = 0

        return thickness
    
    def catchstructure(image,structure):
        #if (structure['name']=='*Skull'):
            #return
        print(structure['name'])
        position=image["imagelocation"]
        pixel = image['pixel']
        #包來自於dicompyler/baseplugins/2dview.py
        if not "zarray" in structure:
            structure['zarray'] = np.array(list(structure['planes'].keys()), dtype=np.float32)
            structure['zkeys'] = structure['planes'].keys()
        # Return if there are no z positions in the structure data
        if not len(structure['zarray']):
            return
        #這邊要去想方法一次抓出全部的座標
        
        structurerange=dicommethodcore.GetContourPixelData(image,structure['datarange'])
        
        structurepointlist={}
        structurepixel={}
        structurepixelshpae=[]
        #listrange={}
        for z in range(0,len(structure['zkeys'])) :
            #?zarray跟position的關係
            zmin = np.amin(np.abs(structure['zarray'][z] - list(map(float, position))))
            index = np.argmin(np.abs(structure['zarray'][z] - list(map(float, position))))
            #zmin = np.amin(np.abs(structure['zarray'] - float(position)))
            #index = np.argmin(np.abs(structure['zarray'] - float(position)))
            
            # Draw the structure only if the structure has contours
            # on the closest plane, within a threshold
            #if (zmin < 0.5):
            if (zmin < 1.5):
                # Set the color of the contour
                '''
                color = wx.Colour(structure['color'][0], structure['color'][1],
                    structure['color'][2], int(self.structure_fill_opacity*255/100))
                # Set fill (brush) color, transparent for external contour
                if (('type' in structure) and (structure['type'].lower() == 'external')):
                    gc.SetBrush(wx.Brush(color, style=wx.TRANSPARENT))
                else:
                    gc.SetBrush(wx.Brush(color))
                gc.SetPen(wx.Pen(tuple(structure['color']),
                    style=self.GetLineDrawingStyle(self.structure_line_style)))
                # Create the path for the contour
                path = gc.CreatePath()
                '''
                for contour in structure['planes'][list(structure['zkeys'])[z]]:
                    if (contour['type'] == u"CLOSED_PLANAR"):
                        
                        # Convert the structure data to pixel data
                        pixeldata = dicommethodcore.GetContourPixelData(
                            image, contour['data'])
                        
                        
                        # fill all corrindonate

                        # Move the origin to the last point of the contour
                        
                        pixeldata=dicommethodcore.connectcircle(pixeldata)
                        #point = pixeldata[-1]
                        #catch all pixelpoint
                        if pixeldata is None:
                            return 
                        pixelpoint= dicommethodcore.fillcorrinate(pixeldata)
                        #shpae pixel
                        #structurepixeldata=np.array([[-1 for y in range(len(sorted(list(set(list(zip(*pixelpoint))[1])))))] for x in range(len(sorted(list(set(list(zip(*pixelpoint))[0])))))],dtype=np.int16)
                        #[number[row[col]]] so if see x should see col as see y should row
                        #plot x  corrindiate see col y corrindiate see row
                        structurepixeldata=np.array([[-1 for x in range(structurerange[0][0],(structurerange[1][0]+1))]for y in range(structurerange[0][1],(structurerange[1][1]+1))],dtype=np.int16)
                        
                        #structurepixeldata=np.array([[-1 for x in range(len(sorted(list(set(list(zip(*pixelpoint))[0])))))]for y in range(len(sorted(list(set(list(zip(*pixelpoint))[1])))))],dtype=np.int16)
                        ida=0
                        listx=list(x-structurerange[0][0] for x in (list(zip(*pixelpoint))[0]))
                        listy=list(y-structurerange[0][1] for y in (list(zip(*pixelpoint))[1]))
                        '''
                        listran=[min(list(zip(*pixelpoint))[0]),max(list(zip(*pixelpoint))[0]),min(list(zip(*pixelpoint))[1]),max(list(zip(*pixelpoint))[1])]
                        listx=list(x-min(list(zip(*pixelpoint))[0]) for x in (list(zip(*pixelpoint))[0]))
                        listy=list(y-min(list(zip(*pixelpoint))[1]) for y in (list(zip(*pixelpoint))[1]))
                        '''
                        for i in range(0,len(listx)):
                            #structurepixeldata[listx[ida]][listy[ida]]=pixel[index][pixelpoint[ida][1]][pixelpoint[ida][0]]
                            structurepixeldata[listy[ida]][listx[ida]]=pixel[index][pixelpoint[ida][1]][pixelpoint[ida][0]]
                            ida=ida+1
                #listrange[(structure['zarray'][z])]=listran
                structurepixelshpae.append(structurepixeldata)
                #'''                
                structurepointlist[(structure['zarray'][z])] = pixelpoint
                structurepixel[(structure['zarray'][z])]=pixel[index][(list(zip(*pixelpoint))[0]),(list(zip(*pixelpoint))[1])]
                #'''    
            #抓出座標，接這去抓rgb
        '''
        if (len(structure['zarray'])>1):
            if (structure['zarray'][0]>structure['zarray'][-1]):
                structurepixelshpae.reverse()
        '''
        structurepixelshpae=np.array(structurepixelshpae,dtype=np.int16)        
        return structurepointlist, structurepixel,structurepixelshpae,structurerange
        #return structurepointlist

    def GetContourPixelData(image, contour):
        """Convert structure data into pixel data using the patient to pixel LUT."""

        pixeldata = []
        pixlut=image['pixellut']
        # Determine whether the patient is prone or supine
        if 'p' in image['patientposition'].lower():
            prone = True
        else:
            prone = False
        # Determine whether the patient is feet first or head first
        if 'ff' in image['patientposition'].lower():
            feetfirst = True
        else:
            feetfirst = False
        # For each point in the structure data
        # look up the value in the LUT and find the corresponding pixel pair
        for p, point in enumerate(contour):
            for xv, xval in enumerate(pixlut[0]):
                if (xval > point[0] and not prone and not feetfirst):
                    break
                elif (xval < point[0]):
                    if feetfirst or prone:
                        break
            for yv, yval in enumerate(pixlut[1]):
                if (yval > point[1] and not prone):
                    break
                elif (yval < point[1] and prone):
                    break
            pixeldata.append((xv, yv))

        return pixeldata
    #抓出全部座標
    def connectcircle(circledata):
        """
        input:
        params circledata: give close data
        output:
        circle:conncet  close circle
        
        """
        if (circledata[0])!=(circledata[-1]):
            print('data not close')
            return
        dataind=0
        circle=[]
        while (dataind < (len(circledata)-1)):
            while (dataind< (len(circledata)-1)):
                datafirst=circledata[dataind]
                dataind=dataind+1
                dataend=circledata[dataind]
                if (((datafirst[0])==(dataend[0]))):
                    circle.append(datafirst)
                else:            
                    break


            circletemp=[]
            if datafirst[0]>dataend[0]:
                startx= dataend[0]
                endx=datafirst[0]
                starty=dataend[1]
                endy=datafirst[1]
                fismen=1
            elif(datafirst[0]<dataend[0]):
                startx= datafirst[0]
                endx=dataend[0]
                starty=datafirst[1]
                endy=dataend[1]
                fismen=0

            #yid=-1
            for x in range(startx, (endx+1)):
                y=((x-startx)*(endy-starty)/(endx-startx))+starty
                if (int(y)!=y):
                    #取上，取下
                    #yid=yid+1
                    circletemp.append((x,int(y)))
                    #circletemp.insert(yid,(int(x),int(y)))   
                else:
                    circletemp.append((x,int(y)))
                    #yid=yid+1
                    #circletemp.insert(yid,(int(x),y)) 

                if (x== startx):
                    #yl=1
                    pass

                else:
                    #if ((circletemp[-1][1])>(circletemp[(yid-yl)][1])):
                    if ((circletemp[-1][1])>(circletemp[(-2)][1])):
                        #slope>0
                        circletempmax=circletemp[-1][1]+1
                        #circletempmin=circle[(yid-yl)][1]+1
                        circletempmin=circletemp[(-2)][1]+1
                    #elif((circletemp[-1][1])<(circletemp[(yid-yl)][1])):
                    elif((circletemp[-1][1])<(circletemp[(-2)][1])):
                        #slope<0
                        #circletempmax=circle[(yid-yl)][1]
                        circletempmax=circletemp[(-2)][1]
                        circletempmin=circletemp[-1][1]
                    else:
                        circletempmax=-1
                        circletempmin=0              
                    yid=0
                    #yl=1 
                    for temp in range(circletempmin,circletempmax):
                        #y fromsmall to big 
                        #circletemp.append(((x-1),(temp)))
                        circletemp.insert((yid-1),((x-1),(temp)))
                        #yl=yl+1
                        yid=yid+1
                        #insert O (n) append O(1) extend O(k)

            #if ((dataind!=0)):
            if (circletemp[0] in circle):
                #remove first,to avoid repat last 
                del circletemp[0]
            if (fismen):
                circle.extend(circletemp[-1::-1])
            else:
                circle.extend(circletemp)
        return circle
    
    #找不到參考文件
    #connect circle點出的點，來給出座標
    def fillcorrinate(pixeldata):
        #xpixlut=pixlut[0]
        #ypixlut=pixlut[1]
        #sortx=sorted(pixeldata, key=lambda x:x[0])
        sortx=sorted(list(set(pixeldata)),key=lambda x:(x[0],x[1]))
        #remove same point and sort b x and y 
        xfirstindex=(min(sortx))[0]
        xlastindex=(max(sortx))[0]
        sortlist=list(zip(*sortx))
        #xfirstindex = np.argmin(np.abs((sortx[0][0])-list(map(float, xpixlut))))
        #xlastindex=np.argmin(np.abs((sortx[-1][0])-list(map(float, xpixlut))))
        findsortdata=[]
        for i in range(xfirstindex,(xlastindex+1)):
            global forwardindfirstx_y,backwardindlastx_y,backwardindfirstx_y,backwardindlastx_y
            #用組數去比對
            if (i in sortlist[0]):
                #while x  in list
                firstx = sortlist[0].index(i)         
                lastx=len(sortx)-(((sortlist[0])))[-1::-1].index(i)-1            
                '''
                firstx=(((list(zip(*sortx)))[0])).index(xpixlut[i])
                firstx_y=np.argmin(np.abs((sortx[firstx][1])-list(map(float, ypixlut))))
                lastx=len(sortx)-(((list(zip(*sortx)))[0]))[-1::-1].index(xpixlut[i])-1
                lastx_y=np.argmin(np.abs((sortx[lastx][1])-list(map(float, ypixlut))))
                '''
                if ((lastx-firstx)):
                    #only in 
                    firstx_y=sortx[firstx][1]
                    lastx_y=sortx[lastx][1]
                    originalsortdata_y=(sortlist[1][firstx:(lastx+1)])
                    for j in range(firstx_y,lastx_y):
                        if j not in originalsortdata_y:
                            findsortdata += [(i,j)]
                else:
                    #while x have same 
                    if ((i  == xfirstindex)):
                        pass
                    elif(i  == (xlastindex)):
                        pass
                    else:
                        forwardind=i+1
                        backwardind=i-1
                        while (forwardind<(xlastindex+1)):
                            if (forwardind in sortlist[0]):
                                forwardindfirstx=sortlist[0].index(forwardind)
                                forwardindlastx=len(sortx)-(((sortlist[0])))[-1::-1].index(forwardind)-1
                                if((forwardindlastx-forwardindfirstx) or (forwardind==(xlastindex))):
                                    forwardindfirstx_y=sortx[forwardindfirstx][1]
                                    forwardindlastx_y=sortx[forwardindlastx][1]
                                    break
                                else:
                                    forwardind=forwardind+1                    
                            else:
                                forwardind=forwardind+1    
                        while (backwardind>(xfirstindex-1)):
                            if (backwardind in sortlist[0]):
                                backwardindfirstx=sortlist[0].index(backwardind)
                                backwardindlastx=len(sortx)-(((sortlist[0])))[-1::-1].index(backwardind)-1
                                if((backwardindlastx-backwardindfirstx) or (backwardind==xfirstindex)):
                                    backwardindfirstx_y=sortx[backwardindfirstx][1]
                                    backwardindlastx_y=sortx[backwardindlastx][1]
                                    break
                                else:
                                    backwardind=backwardind-1
                            else:
                                backwardind=backwardind-1
                        middle_y=sortx[lastx][1]  
                        firstx_y=((forwardindfirstx_y*(i-backwardind))+(backwardindfirstx_y*(forwardind-i)))/(forwardind-backwardind)
                        lastx_y=((forwardindlastx_y*(i-backwardind))+(backwardindlastx_y*(forwardind-i)))/(forwardind-backwardind)
                        if ((middle_y-firstx_y)>(lastx_y-middle_y)):
                            lastx_y=middle_y
                            firstx_y=round(firstx_y)
                        elif ((middle_y-firstx_y)<(lastx_y-middle_y)):
                            firstx_y=middle_y
                            lastx_y=round(lastx_y)

                        if (lastx_y==firstx_y):
                            findsortdata += [(i,firstx_y)]
                        else:
                            for j in range(int(firstx_y),int(lastx_y+1)):
                                findsortdata += [(i,j)]
                        #先找鄰近的位置求內差
            else:
                #while x not in list 
                #first and last must in sortlist
                forwardind=i+1
                backwardind=i-1
                while (forwardind<(xlastindex+1)):
                    if (forwardind in sortlist[0]):
                        forwardindfirstx=sortlist[0].index(forwardind)
                        forwardindlastx=len(sortx)-(((sortlist[0])))[-1::-1].index(forwardind)-1
                        if((forwardindlastx-forwardindfirstx) or (forwardind==(xlastindex))):
                            forwardindfirstx_y=sortx[forwardindfirstx][1]
                            forwardindlastx_y=sortx[forwardindlastx][1]
                            break
                        else:
                            forwardind=forwardind+1                    
                    else:
                        forwardind=forwardind+1       
                while (backwardind>(xfirstindex-1)):

                    if (backwardind in sortlist[0]):
                        backwardindfirstx=sortlist[0].index(backwardind)
                        backwardindlastx=len(sortx)-(((sortlist[0])))[-1::-1].index(backwardind)-1
                        if((backwardindlastx-backwardindfirstx) or (backwardind==xfirstindex)):
                            backwardindfirstx_y=sortx[backwardindfirstx][1]
                            backwardindlastx_y=sortx[backwardindlastx][1]
                            break
                        else:
                            backwardind=backwardind-1
                    else:
                        backwardind=backwardind-1

                firstx_y=((forwardindfirstx_y*(i-backwardind))+(backwardindfirstx_y*(forwardind-i)))/(forwardind-backwardind)
                lastx_y=((forwardindlastx_y*(i-backwardind))+(backwardindlastx_y*(forwardind-i)))/(forwardind-backwardind)
                if (lastx_y==firstx_y):
                    findsortdata += [(i,firstx_y)]
                else:
                    for j in range(int(firstx_y),(int(lastx_y+1))):
                        findsortdata += [(i,j)]

        #findsortdata.append(sortx)
        sortdata=(findsortdata+sortx)
        sortdata=sorted(list(set(sortdata)),key=lambda x:(x[0],x[1]))
        return sortdata

    
    def GetPlan(plandata):
        """Returns the plan information."""
        plan = {}
        for i in range(0,len(plandata)):
            rtplan={}
            data=plandata[i]
            rtplan['label'] = data.RTPlanLabel
            rtplan['date'] = data.RTPlanDate
            rtplan['time'] = data.RTPlanTime
            rtplan['name'] = ''
            rtplan['rxdose'] = 0                  
            rtplan['rtss']=data.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
            rtplan['id']=data.SOPInstanceUID
            if "DoseReferenceSequence" in data:
                for item in data.DoseReferenceSequence:
                    if item.DoseReferenceStructureType == 'SITE':
                        rtplan['name'] = "N/A"
                        if "DoseReferenceDescription" in item:
                            rtplan['name'] = item.DoseReferenceDescription
                        if 'TargetPrescriptionDose' in item:
                            rxdose = item.TargetPrescriptionDose * 100
                            if (rxdose > rtplan['rxdose']):
                                rtplan['rxdose'] = rxdose
                    elif item.DoseReferenceStructureType == 'VOLUME':
                        if 'TargetPrescriptionDose' in item:
                            rtplan['rxdose'] = item.TargetPrescriptionDose * 100
            if (("FractionGroupSequence" in data) and (rtplan['rxdose'] == 0)):
                fg = data.FractionGroupSequence[0]
                if ("ReferencedBeamSequence" in fg) and ("NumberOfFractionsPlanned" in fg):
                    beams = fg.ReferencedBeamSequence
                    fx = fg.NumberOfFractionsPlanned
                    for beam in beams:
                        if "BeamDose" in beam:
                            rtplan['rxdose'] += beam.BeamDose * fx * 100
            rtplan['rxdose'] = round(rtplan['rxdose'])
            rtplan['brachy'] = False
            if ("BrachyTreatmentTechnique" in data) or ("BrachyTreatmentType" in data):
                rtplan['brachy'] = True
            plan[i]=rtplan
        return plan
        #converts raw values into Houndsfeld units
    #CT with RescaleSlope, as MR without
    def get_img_pixels_hu(scans):       
        image = np.stack([s.pixel_array for s in scans])
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
        ConstPixelDims = ( len(scans),int(scans[0].Rows), int(scans[0].Columns))
        # Load spacing values (in mm)
        ConstPixelSpacing = (float(scans[0].SliceThickness),float(scans[0].PixelSpacing[0]), float(scans[0].PixelSpacing[1]))
        x = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
        y = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
        z = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
        #ipp=([(patientall['images'][s].ImagePositionPatient) for s in range(0,len(patientall['images']))])
        #iop=([(patientall['images'][s].ImageOrientationPatient) for s in range(0,len(patientall['images']))])    if hasattr(patient[0], 'RescaleSlope'):

        if hasattr(scans[0], 'RescaleSlope'):
            # Set outside-of-scan pixels to 1
            # The intercept is usually -1024, so air is approximately 0
            image[image == -2000] = 0

            # Convert to Hounsfield units (HU)
            intercept = scans[0].RescaleIntercept
            slope = scans[0].RescaleSlope

            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(np.int16)        
            image += np.int16(intercept)
        else:
            #image = np.zeros(ConstPixelDims, dtype=scans[0].pixel_array.dtype)
            for filenameDCM in scans:
                image[scans.index(filenameDCM),:, :] = filenameDCM.pixel_array    
            image = image.astype(np.int16)
        corridinate=[x,y,z]
        #imagelocation=GetImageLocation(scans)

        return np.array(image, dtype=np.int16),corridinate
    
    def GetPatientToPixelLUT(image):
        """Get the image transformation matrix from the DICOM standard Part 3
            Section C.7.6.2.1.1"""
        self=image
        
        image_x=[]
        image_y=[]
        di = self.PixelSpacing[0]
        dj = self.PixelSpacing[1]
        orientation = self.ImageOrientationPatient
        position = self.ImagePositionPatient

        m = np.matrix(
                [[orientation[0]*di, orientation[3]*dj, 0, position[0]],
                [orientation[1]*di, orientation[4]*dj, 0, position[1]],
                [orientation[2]*di, orientation[5]*dj, 0, position[2]],
                [0, 0, 0, 1]])

        x = []
        y = []
        for i in range(0, self.Columns):
            imat = m * np.matrix([[i], [0], [0], [1]])
            x.append(float(imat[0]))
        for j in range(0, self.Rows):
            jmat = m * np.matrix([[0], [j], [0], [1]])
            y.append(float(jmat[1]))
        image_x=np.array(x)
        image_y=np.array(y)

        return (image_x, image_y)
    
    def GetImageLocation(image):
        """Calculate the location of the current image slice."""
        imagelocation=[]
        for s in range(0,len(image)):
            self=image[s]
            ipp = self.ImagePositionPatient
            iop = self.ImageOrientationPatient

            normal = []
            normal.append(iop[1] * iop[5] - iop[2] * iop[4])
            normal.append(iop[2] * iop[3] - iop[0] * iop[5])
            normal.append(iop[0] * iop[4] - iop[1] * iop[3])

            loc = 0
            for i in range(0, len(normal)):
                loc += normal[i] * ipp[i]

        # The image location is inverted for Feet First images
            if 'PatientPosition' in self:
                patientposition=self.PatientPosition.lower()
                if ('ff' in self.PatientPosition.lower()):
                    loc = loc * -1
                
                    
            imagelocation += [loc]
        return imagelocation,patientposition
    def MultiGetPatientToPixelLUT(image):
        """Get the image transformation matrix from the DICOM 
        standard Part 3 Section C.7.6.2.1.1"""
        image_x=[]
        image_y=[]
        for s in range(0,len(image)):
            self=image[s]
            di = self.PixelSpacing[0]
            dj = self.PixelSpacing[1]
            orientation = self.ImageOrientationPatient
            position = self.ImagePositionPatient

            m = np.matrix(
                [[orientation[0]*di, orientation[3]*dj, 0, position[0]],
                [orientation[1]*di, orientation[4]*dj, 0, position[1]],
                [orientation[2]*di, orientation[5]*dj, 0, position[2]],
                [0, 0, 0, 1]])

            x = []
            y = []
            for i in range(0, self.Columns):
                imat = m * np.matrix([[i], [0], [0], [1]])
                x.append(float(imat[0]))
            for j in range(0, self.Rows):
                jmat = m * np.matrix([[0], [j], [0], [1]])
                y.append(float(jmat[1]))
            image_x+=[np.array(x)]
            image_y+=[np.array(y)]

        return (image_x, image_y)
    
    def doseimg(image,dose, threshold=0.5,level=[30,50,60,70,80,90,100]):
        """Input
        :param threshold:  Threshold in mm to determine the max difference
                              from z to the closest dose slice without
                              using interpolation.
        :param level:       Isodose level in scaled form
        :param isodose:    the Tc dose
        """
        """Output:
        :param doseplane     the data of dose
        :param IsodosePoints level of dose indicates with  pixel lut
        :param dosemax      max dose
        :param corridinate  pixel lut
        :param dosegridscaling dose grid level
        
        """
        self=dose
        """Convert dosegrid data into pixel data using the dose to pixel LUT."""
        structurepixlut=(image['pixellut'])
        position=image["imagelocation"]
        doselut=dicommethodcore.GetPatientToPixelLUT(dose)
        x = []
        y = []
        # Determine if the patient is prone or supine 
        prone = -1 if 'p' in image['patientposition'].lower() else 1
        feetfirst = -1 if 'ff' in image['patientposition'].lower() else 1
        # Get the pixel spacing
        spacing = [(abs(structurepixlut[0][0]-structurepixlut[0][1])),(abs(structurepixlut[1][0]-structurepixlut[1][1]))]

        # Transpose the dose grid LUT onto the image grid LUT
        x = (np.array(doselut[0]) - structurepixlut[0][0]) * prone * feetfirst / spacing[0]
        y = (np.array(doselut[1]) - structurepixlut[1][0]) * prone / spacing[1]
        
        '''
        xpos,ypos :you indicates x,y position
            xdpos = np.argmin(np.fabs(np.array(x) - xpos))
            ydpos = np.argmin(np.fabs(np.array(y) - ypos))
                dosegrid = dose.GetDoseGrid(float(self.z))
                if not (dosegrid == []):
                    dose = dosegrid[ydpos, xdpos] * \
                           dosegridscaling
        '''
        dosegridscaling=self.DoseGridScaling
        #GetDoseGrid
        if 'GridFrameOffsetVector' in self:
            #GetDoseGrid
            # Get the initial dose grid position (z) in patient coordinates
            imagepatpos = self.ImagePositionPatient[2]
            orientation = self.ImageOrientationPatient[0]
            # Add the position to the offset vector to determine the
                # z coordinate of each dose plane
            planes = orientation * np.array(self.GridFrameOffsetVector) + imagepatpos
            #maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
            dosemax =int(float(dose.pixel_array.max())* self.DoseGridScaling* 100)
            #dosemax = int(dosedata['dosemax'] * dosedata['dosegridscaling'] * 10000 / plan['rxdose'])
            #dosemax=int(float(dose.pixel_array.max()) * self.DoseGridScaling * 10000 / isodose)
            
            
            IsodosePoints={}
            doseplane={}
            
            for i in range(0,len(position)):
                z = float(position[i])
                if(((np.amax(position))>= z >=(np.amin(position)))):
                    #only find range if and only if z bwtween position
                    frame = -1
                    # Check to see if the requested plane exists in the array
                    if (np.amin(np.fabs(planes - z)) < threshold):
                        frame = np.argmin(np.fabs(planes - z))
                    # Return the requested dose plane, since it was found
                    if not (frame == -1):
                        plane= self.pixel_array[frame]
                        doseplane[z]=plane
                        # Check if the requested plane is within the dose grid boundaries
                    elif ((z < np.amin(planes)) or (z > np.amax(planes))):
                        plane= np.array([])
                        doseplane[z]=plane
                    # The requested plane was not found, so interpolate between planes
                    else:
                    # Determine the upper and lower bounds
                        umin = np.fabs(planes - z)
                        ub = np.argmin(umin)
                        lmin = umin.copy()
                    # Change the min value to the max so we can find the 2nd min
                        lmin[ub] = np.amax(umin)
                        lb = np.argmin(lmin)
                    # Fractional distance of dose plane between upper & lower bound
                        fz = (z - planes[lb]) / (planes[ub] - planes[lb])
                        plane = fz*self.pixel_array[ub] + (1.0 - fz)*self.pixel_array[lb]
                        doseplane[z]=plane
                    #'''
                    #GetIsodosePoints
                    
                    for j in range(0,len(level)):
                        # Calculate the isodose level according to rx dose and dose grid scaling
                        #level = isodose['data']['level'] * self.rxdose / (self.dosedata['dosegridscaling'] * 10000)
                        doselevel = level[j] *  dosemax / (self.DoseGridScaling  * 10000)
                        # this level indicate the true dose value 
                        isodose = (plane >= doselevel)
                        if  (len(isodose)):
                            isodose=isodose.nonzero()
                            point=([(isodose[1].tolist(), isodose[0].tolist())])
                            pointlist=[]
                            for x,y in point:
                                pointlist +=(list(zip(doselut[0][x],doselut[1][y])))
                            if (len(pointlist)):
                                if not z in IsodosePoints :
                                    IsodosePoints[z]={} 
                                IsodosePoints[z][level[j]]= pointlist
                            #IsodosePoints[z][level[j]] = (list(zip(isodose[1].tolist(), isodose[0].tolist())))
                            
                    #from IsodosePoints indices level percent point to corrindinate which in the x-th,y-th point 
                    #'''
                imgdosecorridinate=[x,y]
            return doseplane,dosemax,doselut,imgdosecorridinate,dosegridscaling,IsodosePoints    
        else:
            return np.array([])
        
class dicom_read:
    def load_scan(path):
        files = dicommethodcore.searchpath(path)
        patients,h_record,ptdata=dicommethodcore.getdicomdatainfo(files)
        image={}
        structure={}
        rtdose={}
        plan={}
        patientinfo={}
        for s1 in h_record:
            for s2 in h_record[s1]:
                if not s1 in patientinfo:
                    patientinfo[s1]={}
                patientinfo[s1][s2]=patients[s1][s2]['patientinfo']
                for si in h_record[s1][s2]:
                    if 'images' in ptdata[s1][s2]:
                        if not s1 in image:
                            image[s1]={}
                        if not s2 in image[s1]:
                            image[s1][s2]={}
                        if not si in image[s1][s2]:
                            image[s1][s2][si]={}
                        ptdata[s1][s2]['images'][si]=dicommethodcore.resortimage(ptdata,s1,s2,si)
                        pixel,corridinate=dicommethodcore.get_img_pixels_hu(ptdata[s1][s2]['images'][si])
                        image[s1][s2][si]['pixel']=pixel
                        image[s1][s2][si]['corridinate']=corridinate
                        imagelocation,patientposition=dicommethodcore.GetImageLocation(ptdata[s1][s2]['images'][si])
                        image[s1][s2][si]['imagelocation']=imagelocation
                        image[s1][s2][si]['patientposition']=patientposition
                        image[s1][s2][si]['pixellut']=dicommethodcore.GetPatientToPixelLUT(ptdata[s1][s2]['images'][si][0])

                    if 'structures' in ptdata[s1][s2]:
                        if not s1 in structure:
                            structure[s1]={}
                        if not s2 in structure[s1]:
                            structure[s1][s2]={}
                        if not si in structure[s1][s2]:
                            structure[s1][s2][si]={}
                        d = ptdata[s1][s2]['structures'][si]
                        st = dicommethodcore.structure_method(d)
                        for i in range(0,len(d)):
                            for k in st[i].keys():
                                if (st[i][k]['name']!='*Skull'):
                                    structureplanes,datarange=dicommethodcore.GetStructureCoordinates(d[i],k)
                                    st[i][k]['datarange']=datarange
                                    st[i][k]['planes'] = structureplanes
                                    st[i][k]['thickness'] = dicommethodcore.CalculatePlaneThickness(d[i],st[i][k]['planes'])
                                    planepoint=dicommethodcore.catchstructure(image[s1][s2][si],st[i][k])
                                    #planepoint,planepixel,structurepixelshpae,structurerange=dicommethodcore.catchstructure(image[s1][s2][si],st[i][k])

                                    #planepoint=dicommethodcore.catchstructure(image[s][si],st[i][k])
                                    if planepoint is not None:
                                        st[i][k]['planepoint']=planepoint[0]
                                        st[i][k]['planepixel']=planepoint[1]
                                        st[i][k]['structurepixelshpae']=planepoint[2]
                                        st[i][k]['structurerange']=planepoint[3]
                                        #st[i][k]['listrange']=listrange

                        structure[s1][s2][si] = st
                # plans、
                if 'plans' in ptdata[s1][s2]:
                    if not s1 in plan:
                        plan[s1]={}
                    plan[s1][s2] = dicommethodcore.GetPlan(ptdata[s1][s2]['plans'])
                    for j in range(0,len(h_record[s1][s2])):
                        if len(plan[s1][s2][j]['rtss']):
                            for x ,y in enumerate(patients[s1][s2]['structures']):
                                if ((plan[s1][s2][j]['rtss'])== y):
                                    plan[s1][s2][j]['imgseries']=patients[s1][s2]['structures'][y]['series']
                    if 'doses' in ptdata[s1][s2]:
                        for j in range(0,len(h_record[s1][s2])):
                            for x,y in enumerate((patients[s1][s2]['doses'])):
                                if ((plan[s1][s2][j]['id']) == (patients[s1][s2]['doses'][y]['rtplan'])):
                                    if not 'doses' in plan[s1][s2][j]:
                                        plan[s1][s2][j]['doses']=[]
                                    plan[s1][s2][j]['doses'] +=[y]

                if 'doses' in ptdata[s1][s2]:
                    if not s1 in rtdose:
                        rtdose[s1]={}
                    if not s2 in rtdose[s1]:
                        rtdose[s1][s2]={}
                    j=0
                    for i in patients[s1][s2]['doses']:
                        for k in  range(0,len(h_record[s1][s2])):
                            if ((ptdata[s1][s2]['doses'][j].ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID)==
                                (plan[s1][s2][k]['id'])):
                                si = plan[s1][s2][k]['imgseries']
                            if not si in rtdose[s1][s2]:
                                rtdose[s1][s2][si]={}
                        if not'dvhs'in rtdose[s1][s2][si]:
                            rtdose[s1][s2][si]['dvhs']={}
                            rtdose[s1][s2][si]['doseplane']={}
                            rtdose[s1][s2][si]['IsodosePoints']={}
                            rtdose[s1][s2][si]['dosemax']={}
                            rtdose[s1][s2][si]['pixellut']={}
                            rtdose[s1][s2][si]['imgdosecorridinate']={}
                            rtdose[s1][s2][si]['dosegridscaling']={}
                        if patients[s1][s2]['doses'][i]['hasdvh']:
                            DVH= dicommethodcore.GetDVHs(ptdata[s1][s2]['doses'][j])
                            doseplane,dosemax,doselut,imgdosecorridinate,dosegridscaling= dicommethodcore.doseimg(image[s1][s2][si],ptdata[s1][s2]['doses'][j],DVH)
                            rtdose[s1][s2][si]['dvhs'][i]=DVH
                        else:
                            doseplane,dosemax,doselut,imgdosecorridinate,dosegridscaling,IsodosePoints = dicommethodcore.doseimg(image[s1][s2][si],ptdata[s1][s2]['doses'][j])
                            #套運rtdose method
                        rtdose[s1][s2][si]['doseplane'][i]=doseplane
                        rtdose[s1][s2][si]['IsodosePoints'][i]=IsodosePoints
                        rtdose[s1][s2][si]['dosemax'][i]=dosemax
                        rtdose[s1][s2][si]['pixellut'][i]=doselut
                        rtdose[s1][s2][si]['imgdosecorridinate'][i]=imgdosecorridinate
                        rtdose[s1][s2][si]['dosegridscaling'][i]=dosegridscaling
                        j=j+1


        return patientinfo, h_record,image,structure,plan,rtdose





