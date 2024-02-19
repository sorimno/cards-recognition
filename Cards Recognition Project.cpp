#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include <algorithm>
#include <fstream>
#include <stdio.h>

using namespace std;

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2);			//Function(FLAG) for sorting contours by Area
void contoursApproximation(vector<vector<cv::Point>>& input, vector<vector<cv::Point>>& output);	//Aproximating the contours
void cardsTransformation(vector<cv::Point>& punkty, cv::Mat& input, cv::Mat& output);				//Perspective transformation function
void contoursFiltration(vector<vector<cv::Point>> kontury[4], vector<vector<double>>& area, int i);	//Filtration of undesirable contours
void sortContours(vector<vector<cv::Point>> kontury[4], vector<vector<double>>& area, int i);		//Sorting vector of contours
void findMoments(vector<vector<cv::Point>> kontury[4], vector<vector<cv::Moments>>& mom, int i);	//Calculating Moments
void findHuMoments(vector<vector<cv::Moments>>& mom, vector<vector<vector<double>>>& hu, int i);	//Calculating HuMoments
void cardRecognition(vector<vector<double>> input, int& ID);
void suitsRecognition(vector<double> input, int& ID);												
bool is_trefl(vector<double> input, int& ID);
bool is_pik(vector<double> input, int& ID);
bool is_kier(vector<double> input, int& ID);
bool is_karo(vector<double> input, int& ID);
void compare_the_sum(vector<int[2]>& input);														//Function comparing the sum of values on cards




int main()

{
	cv::String path("data/*.png");							//Folder containing the pictures for analysis
	vector<cv::String> pictures;							//Vector containing directories of pictures in the folder
	fstream data;										 
	data.open("data/data.txt", ios::out | ios::app);		//Text file to save HuMoments of figures

	cv::glob(path, pictures, true);							//Generates a list of all files that match the globbing pattern

	for (size_t k = 0; k < pictures.size(); ++k)
	{	
		cv::Mat img = cv::imread(pictures[k], 1);			//Input image
		cv::Mat img1(img.size(), img.type());			

		vector<cv::Mat> cardsMasks;							//Vector containing masks of cards
		vector<cv::Vec4i> hierarchy;

		vector<cv::Mat> hsvCh;								//vector containg HSV chanels of input image
		vector<vector<cv::Point>> contours;					//vector of vector containg Points of CONTURS 
		vector<vector<cv::Point>> poly;						//vector of vector containg Points of vertices
		vector<vector<cv::Point>> kontury[4];				
		vector<cv::Mat> cards(4);							
		vector<cv::Mat> originals(4);			
		vector<cv::Mat> oneCardVec;			
		vector<vector<double>> area;
		vector<vector<cv::Moments>> mom;
		vector<vector<vector<double>>> hu;
		vector<int[2]> ID_all_cards(4);						//Outcome vector where ID_all_cards[card#][(value)(suit)]
		
		cv::Mat shadow_mask(img.size(), CV_8U, cv::Scalar(255));
		cv::Mat shadow_mask1(shadow_mask.size(), CV_8U, cv::Scalar(255));
		cv::namedWindow("karta", CV_WINDOW_KEEPRATIO);
		
		cv::medianBlur(img, img, 5);						//median blurring to get rid of noise
		cv::cvtColor(img, img1, CV_BGR2HSV);				//split to HSV to extract shadow mask
		cv::split(img1, hsvCh);
		cv::normalize(hsvCh[1], hsvCh[1], 0, 255, cv::NORM_MINMAX);
		cv::normalize(hsvCh[2], hsvCh[2], 0, 255, cv::NORM_MINMAX);
		shadow_mask = hsvCh[2] - hsvCh[1];
		cv::threshold(shadow_mask, shadow_mask, 20, 255, CV_THRESH_BINARY);
		cv::findContours(shadow_mask, poly, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		sort(poly.begin(), poly.end(), compareContourAreas);
		poly.erase(poly.begin(), poly.end() - 4); //Erasing smaller contours

		for (int i = 0; i < poly.size(); i++)  
		{
			cv::Mat localMat = cv::Mat::zeros(img.size(), img.type());
			cv::Scalar color(255, 255, 255);
			cv::drawContours(localMat, poly, i, color, CV_FILLED, 8);
			cardsMasks.push_back(localMat);
		}

		for (int i = 0; i < cardsMasks.size(); i++) //Finding cards on an image
		{
			cv::Mat localMat = cv::Mat::zeros(img.size(), img.type());
			cv::bitwise_and(img, cardsMasks[i], localMat);
			oneCardVec.push_back(localMat);
		}

		contoursApproximation(poly, contours);

		for (int i = 0; i < 4; ++i)				//Transformating a card
		{
			cardsTransformation(contours[i], oneCardVec[i], cards[i]); 
		}
		for (int i = 0; i < cards.size(); i++)	//Copying the transformed card to a new vector of Mats
		{
			cards[i].copyTo(originals[i]);
		}

		for (int i = 0; i < cards.size(); i++) //Threshodling a card
		{
			
			cv::cvtColor(cards[i], cards[i], CV_BGR2GRAY);
			cv::medianBlur(cards[i], cards[i], 5);
			cv::threshold(cards[i], cards[i], 100, 255, CV_THRESH_BINARY);
	
		}

		
		for (int i = 0; i < 4; i++)			//Finding contours on a card
		{
			cv::Mat kontur(cv::Size(500, 500), CV_8U, cv::Scalar(255));
			cv::findContours(cv::Mat(cards[i]), kontury[i], cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
			cv::polylines(kontur, kontury[i], true, 0, 2);
		}


		for (int i = 0; i < 4; ++i)			//Sorting contours by an area
		{
			sortContours(kontury, area, i);
		}

		for (int i = 0; i < 4; ++i)			//Filtration of conturs 
		{
			cv::Mat kontur(cv::Size(500, 500), CV_8U, cv::Scalar(255));
			contoursFiltration(kontury, area, i);
			cv::polylines(kontur, kontury[i], true, 0, 2);
		}

		for (int i = 0; i < 4; i++)			//Calculation Moments
		{
			findMoments(kontury, mom, i);
		}

		for (int i = 0; i < 4; i++)			//Calculation HuMoments
		{
			findHuMoments(mom, hu, i);
		}
		string name;
		cout << "This is a picture #" << k << endl;
		for (int i = 0; i < 4; i++)			//Recognition of a card
		{
			ID_all_cards[i][0] = hu[i].size();
			cv::Mat kontur(cv::Size(500, 500), CV_8U, cv::Scalar(255));
			cout << "Card has a value "<<hu[i].size()<<" ";
			cardRecognition(hu[i], ID_all_cards[i][1]);
			imshow("karta", originals[i]);
			switch (ID_all_cards[i][1])
			{
			case 0:
				cout << "TREFL (Clovers) ";
				break;
			case 1:
				cout << "PIK (Pikes)  ";
				break;
			case 2:
				cout << "KIER (Hearts) ";
				break;
			case 3:
				cout << "KARO (Tiles) ";
				break;
			}
			cout << endl;

			cv::waitKey(1200);
			
		}
		compare_the_sum(ID_all_cards);
		cout<<endl<<endl;
		
		/*for (int i = 0; i < 4; i++)
		{
			int suit;
			cv::imshow("karta", cards[i]);
			cout << '\n';
			cout << "1 -- trefl 2 -- pik 3--kier 4--karo" << endl;
			cout << "Please enter what suit is visible on the card. ";
			cv::waitKey(500);
			cin >> suit;

			for (int z = 0; z < hu[i].size(); z++)
			{
				switch (suit)
				{
				case 1:
					data << "trefl:,";
					break;
				case 2:
					data << "pik:,";
					break;
				case 3:
					data << "kier:,";
					break;
				case 4:
					data << "karo:,";
					break;
				}
				for (int y = 0; y < hu[i][z].size(); y++)
				{
					data << hu[i][z][y] << ',';
				}
				data << endl;
			}
		}
		*/ //HuMoments to text file export loop
	}

	cv::waitKey(0);
	return 0;

}


bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) //function comparing Areas of contour
{
	double i = contourArea(cv::Mat(contour1));
	double j = contourArea(cv::Mat(contour2));
	return (i < j);
}

void contoursApproximation(vector<vector<cv::Point>>& input, vector<vector<cv::Point>>& output)
{
	std::vector<std::vector<cv::Point>>::iterator it = input.begin();
	while (it != input.end())
	{
		std::vector<cv::Point2i> temp;
		cv::approxPolyDP(*it, temp, 30, true);	//Approximate by a poligon
		if (temp.size() == 4) {					//Does poligon has only 4 corners?
			output.push_back(temp);
		}
		++it;
	}
}

void cardsTransformation(vector<cv::Point>& punkty, cv::Mat& input, cv::Mat& output)
{
	cv::Mat card;
	vector<cv::Point2f>src_points{punkty[0], punkty[1],punkty[2], punkty[3]};								//Points on the source image							
	vector<cv::Point2f>dst_points{cv::Point(500,0), cv::Point(0,0),cv::Point(0, 500), cv::Point(500,500)};	//Destination points

	cv::Mat T = getPerspectiveTransform(src_points, dst_points);											//Obtaining Transformation matrix
	cv::warpPerspective(input, card, T, cv::Size(500, 500));												//Perspective transformation
	card.copyTo(output);																					//Copying card to the output
}

void contoursFiltration(vector<vector<cv::Point>> kontury[4], vector<vector<double>>& area, int i)
{
	{
		double sumAreas = 0;
		vector<double> localAreas;
		for (int c = 0; c < kontury[i].size(); ++c)
		{
			if ((area[i][c] > 240000) || (area[i][c] < 50)) //erase all contours bigger than 240000 and smaller than 50
			{
				kontury[i].erase(kontury[i].begin() + c);
				area[i].erase(area[i].begin() + c);
				c--;
			}
		}
		for (int c = 0; c < kontury[i].size(); ++c)
		{
			sumAreas += area[i][c];
		}
		double avgArea = sumAreas*0.79 / area[i].size();
		for (int c = 0; c < kontury[i].size(); ++c)
		{
			if (area[i][c] < avgArea)
			{
				kontury[i].erase(kontury[i].begin() + c);
				area[i].erase(area[i].begin() + c);
				c--;
			}
		}
	}
}

void sortContours(vector<vector<cv::Point>> kontury[4], vector<vector<double>>& area, int i)
{
	vector<double> localAreas;
	sort(kontury[i].begin(), kontury[i].end(), compareContourAreas);
	for (auto& c : kontury[i])
	{
		localAreas.push_back(cv::contourArea(c));
	}
	area.push_back(localAreas);
}

void findMoments(vector<vector<cv::Point>> kontury[4], vector<vector<cv::Moments>>& mom, int i)
{
	vector<cv::Moments>localMoment;
	for (auto& c : kontury[i])
	{
		localMoment.push_back(cv::moments(c, false));
	}
	mom.push_back(localMoment);
}

void findHuMoments(vector<vector<cv::Moments>>& mom, vector<vector<vector<double>>>& hu, int i)
{
	vector<vector<double>> localHu;
	for (auto& c :mom [i])
	{
		vector<double> minlocalHu;
		cv::HuMoments(c, minlocalHu);
		localHu.push_back(minlocalHu);
	}
	hu.push_back(localHu);
}

void cardRecognition(vector<vector<double>> input, int& ID)
{
	int trefl = 0, pik = 0, karo = 0, kier = 0;
	int temp_ID;
	for (int i = 0; i < input.size(); i++)
	{
		suitsRecognition(input[i], temp_ID);
		switch (temp_ID)
		{
		case 0:
			trefl++;
			break;
		case 1:
			pik++;
			break;
		case 2:
			kier++;
			break;
		case 3:
			karo++;
			break;
		}
	}

	if (trefl > pik && trefl > kier &&  trefl> karo)	//trefl
		ID = 0;
	if (pik > trefl && pik>kier && pik>karo)			//pik
		ID = 1;
	if (kier > trefl && kier > pik && kier > karo)		//kier
		ID = 2;
	if (karo > trefl && karo > pik && karo > kier)		//karo
		ID = 3;

}

void suitsRecognition(vector<double> input, int& ID)
{
	if (is_trefl(input, ID))
		ID = 0;
	else if (is_pik(input, ID))
		ID = 1;
	else if (is_kier(input, ID))
		ID = 2;
	else if (is_karo(input, ID))
		ID = 3;
}

bool is_trefl(vector<double> input, int& ID)
{
	int count = 0;
	if (input[0] >= 0.184682 && input[0] <= 0.19834)
		count++;
	if (input[1] >= 0.00504859 && input[1] <= 0.00781185)
		count++;
	if (input[2] >= 0.000187789 && input[2] <= 0.000721408)
		count++;
	if (input[3] >= 8.51221E-06 && input[3] <= 2.83261E-05)
		count++;
	if (input[4] >= -3.0329E-09 && input[4] <= -4.46974E-10)
		count++;
	if (input[5] >= -2.25369E-06 && input[5] <= - 6.26869E-07)
		count++;
	if (input[6] >= -1.05162E-09 && input[6] <= 7.92177E-10)
		count++;
	
	if (count == 7)
			return true;

		else
			return false;
}

bool is_pik(vector<double> input, int& ID)
{
	int count = 0;
	if (input[0] >= 0.178198 && input[0] <= 0.193462)
		count++;
	if (input[1] >= 0.00214726 && input[1] <= 0.00594585)
		count++;
	if (input[2] >= 2.18026E-06 && input[2] <= 9.03791E-05)
		count++;
	if (input[3] >= 2.43401E-06 && input[3] <= 1.98446E-05)
		count++;
	if (input[4] >= -3.80043E-10 && input[4] <= -1.25653E-11)
		count++;
	if (input[5] >= -1.16689E-06 && input[5] <= -1.2996E-07)
		count++;
	if (input[6] >= -1.74095E-10 && input[6] <= 1.33665E-10)
		count++;

	if (count == 7)
		return true;

	else
		return false;
}

bool is_kier(vector<double> input, int& ID)
{
	int count = 0;
	if (input[0] >= 0.190624 && input[0] <= 0.204293)
		count++;
	if (input[1] >= 0.00494656 && input[1] <= 0.00911117)
		count++;
	if (input[2] >= 0.00188472 && input[2] <= 0.00273882)
		count++;
	if (input[3] >= 3.39829E-05 && input[3] <= 9.89787E-05)
		count++;
	if (input[4] >= -5.06942E-08 && input[4] <= -7.99416E-09)
		count++;
	if (input[5] >= -9.41606E-06 && input[5] <= -2.25231E-06)
		count++;
	if (input[6] >= -1.91983E-08 && input[6] <= 1.63297E-08)
		count++;

	if (count == 7)
		return true;

	else
		return false;
}

bool is_karo(vector<double> input, int& ID)
{
	int count = 0;
	if (input[0] >= 0.161633 && input[0] <= 0.168998)
		count++;
	if (input[1] >= 2.4593E-06 && input[1] <= 0.000568472)
		count++;
	if (input[2] >= 4.75251E-25 && input[2] <= 0.000116918)
		count++;
	if (input[3] >= 2.29438E-25 && input[3] <= 9.9652E-07)
		count++;
	if (input[4] >= -8.59034E-12 && input[4] <= 9.98144E-13)
		count++;
	if (input[5] >= -3.12298E-09 && input[5] <= 1.44589E-08)
		count++;
	if (input[6] >= -1.03972E-12 && input[6] <= 4.89285E-12)
		count++;

	if (count == 7)
		return true;

	else
		return false;
}

void compare_the_sum(vector<int[2]>& input)
{
	int red = 0, black = 0;
	for (int i = 0; i < input.size(); i++)
	{
		if (input[i][1] == 0 || input[i][1] == 1)// 0 - trefl, 1 - pik
		{
			black += input[i][0];
		}
		if (input[i][1] == 2 || input[i][1] == 3)// 2 - kier, 3 - karo
		{
			red += input[i][0];
		}
	}

	if (red > black)
		cout << "There are more red values." << endl;
	if (black > red)
		cout << "There are more black values." << endl;
	if (red == black)
		cout << "Values on black cards and red cards are equal." << endl;
}
