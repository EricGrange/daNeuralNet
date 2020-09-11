unit FMNIST;

{$i daNN.inc}

interface

uses
  Winapi.Windows, Winapi.Messages, System.SysUtils, System.Variants, System.Classes, Vcl.Graphics,
  Vcl.Controls, Vcl.Forms, Vcl.Dialogs, System.Zip, System.Threading, System.Diagnostics,
  daNeuralNet, daNeuralNet.Dense,
  VclTee.TeeGDIPlus, VCLTee.TeEngine, VCLTee.Series, Vcl.ExtCtrls,
  VCLTee.TeeProcs, VCLTee.Chart, Vcl.StdCtrls;

type
  TFormMNIST = class(TForm)
    BUOneEpoch: TButton;
    ListBox: TListBox;
    Image: TImage;
    Splitter1: TSplitter;
    Chart1: TChart;
    Label1: TLabel;
    BUReset: TButton;
    Series1: TAreaSeries;
    TeeGDIPlus1: TTeeGDIPlus;
    procedure FormCreate(Sender: TObject);
    procedure BUOneEpochClick(Sender: TObject);
    procedure ListBoxClick(Sender: TObject);
    procedure BUResetClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
    FTrainImages, FTestImages : TdaNNDataset;
    FTrainLabels, FTestLabels : TBytes;
    FTrainOutputs, FTestOutputs : TdaNNDataset;
    FLoadTask : ITask;
    FNeuralNet : TdaNeuralNet;
    procedure LoadDatasets;
  end;

var
  FormMNIST: TFormMNIST;

implementation

{$R *.dfm}

procedure TFormMNIST.BUResetClick(Sender: TObject);
begin
   FNeuralNet.RandomizeWeights;
   BUOneEpochClick(Sender);
end;

procedure TFormMNIST.FormCreate(Sender: TObject);
begin
   FLoadTask := TTask.Run(
      procedure
      begin
         LoadDatasets
      end
   );

   FNeuralNet := TdaNeuralNet.Create;
   FNeuralNet.AddLayer(TdaNNInputLayer.Create(28*28));
   FNeuralNet.AddLayer(TdaNNDenseLayer.Create(60, daNNActivation.ReLu, 0.02, 0));
   FNeuralNet.AddLayer(TdaNNDenseLayer.Create(10, daNNActivation.Sigmoid, 0.2, 0));
   FNeuralNet.Build([nnboForTraining]);
   FNeuralNet.RandomizeWeights;
end;

procedure TFormMNIST.BUOneEpochClick(Sender: TObject);
begin
   FLoadTask.Wait;
   ListBox.Clear;

   var sw := TStopwatch.StartNew;
   FNeuralNet.TrainSet(FTrainImages, FTrainOutputs);
   var t := sw.ElapsedMilliseconds;

   ListBox.Items.BeginUpdate;

   var errors := 0;
   for var i := 0 to High(FTestImages) do begin
      var output := FNeuralNet.Run(FTestImages[i]);
      var digit := 0;
      for var k := 1 to High(output) do
         if output[k] > output[digit] then
            digit := k;
      if digit <> FTestLabels[i] then begin
         Inc(errors);
         if errors <= 100 then begin
            ListBox.Items.AddObject(
               Format('For #%d, expected %d but found %d',
                      [ i, FTestLabels[i], digit ]),
               Pointer(i)
            );
         end;
      end;
   end;

   ListBox.Items.EndUpdate;

   Label1.Caption := Format('%.03f sec', [ t * 0.001 ]) + #13#10
                   + Format('Errors : %d  ( %0.2f %% )', [ errors, 100*errors/Length(FTestImages) ]);
   Series1.Add(errors);

   if errors > 0 then begin
      ListBox.ItemIndex := 0;
      ListBoxClick(nil);
   end;
end;

// LoadDatasets
//
procedure TFormMNIST.ListBoxClick(Sender: TObject);
begin
   if ListBox.ItemIndex < 0 then Exit;
   var i := Integer(ListBox.Items.Objects[ListBox.ItemIndex]);
   var bmp := TBitmap.Create;
   try
      bmp.SetSize(28, 28);
      for var y := 0 to bmp.Height-1 do for var x := 0 to bmp.Width-1 do
         bmp.Canvas.Pixels[x, y] := Round(FTestImages[i][y*28 + x]*255) * $10101;
      Image.Picture.Graphic := bmp;
   finally
      bmp.Free;
   end;
end;

procedure TFormMNIST.LoadDatasets;

   procedure LoadImages(var bytes : TBytes; var dataset : TdaNNDataset);
   begin
      var p := 16;
      var nb := (Length(bytes)-p) div (28*28);
      SetLength(dataset, nb);
      for var i := 0 to nb - 1 do begin
         SetLength(dataset[i], 28*28);
         var img := dataset[i];
         for var j := 0 to 28*28-1 do
            img[j] := bytes[p+j] / 255;
         Inc(p, 28*28);
      end;
   end;

   procedure LoadLabels(var bytes : TBytes; var labels : TBytes; var dataset : TdaNNDataset);
   begin
      var p := 8;
      var nb := Length(bytes)-p;
      SetLength(dataset, nb);
      SetLength(labels, nb);
      for var i := 0 to nb - 1 do begin
         labels[i] := bytes[i+8];
         SetLength(dataset[i], 10);
         dataset[i][labels[i]] := 1;
      end;
   end;

begin
   var buf : TBytes;
   var zip := TZipFile.Create;
   try
      zip.Open('..\..\..\Datasets\MNIST-digits\mnist.zip', zmRead);

      zip.Read('train-images.idx3-ubyte', buf);
      LoadImages(buf, FTrainImages);
      zip.Read('train-labels.idx1-ubyte', buf);
      LoadLabels(buf, FTrainLabels, FTrainOutputs);

      zip.Read('t10k-images.idx3-ubyte', buf);
      LoadImages(buf, FTestImages);
      zip.Read('t10k-labels.idx1-ubyte', buf);
      LoadLabels(buf, FTestLabels, FTestOutputs);

   finally
      zip.Free;
   end;
end;

end.
