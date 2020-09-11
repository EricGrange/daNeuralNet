program MNIST_digits;

uses
  Vcl.Forms,
  FMNIST in 'FMNIST.pas' {FormMNIST};

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TFormMNIST, FormMNIST);
  Application.Run;
end.
