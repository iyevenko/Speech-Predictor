import {Component, ElementRef, HostListener, ViewChild} from '@angular/core';
import {TextService} from './text.service';
import {MatInputModule} from '@angular/material/input';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'client';
  playerName;
  responseText;
  public searchStr = '';
  options: string[];

  constructor(private textService: TextService) {
  this.playerName = '';
  this.responseText = '';
  this.options = [];
  console.log('options: ');
  console.log(this.options);

  }

  @ViewChild('textInput') textInput: ElementRef | undefined;

  @HostListener('document:click', ['$event'])

  // documentClick(event: MouseEvent) {
  //   console.log(this.searchStr);
  //   if (!(this.textInput == undefined)){
  //     this.textInput.nativeElement.select() as HTMLInputElement;
  //   }
  // }

submitText() {
    let wordArray = this.searchStr.split(' ');
    let sequence = wordArray.slice(-6).join(" ");
    console.log('Sent: ' + sequence);
    this.textService.submitText(sequence).subscribe((response: any) => {
    this.options = response.response;
      const index: number = this.options.indexOf("[UNK]");
      if (index !== -1) {
        this.options.splice(index, 1);
      }
      this.options = this.options.map(x => this.searchStr + x);

    });
  }

  public modelChange(str: string): void {
    if (str.substr(str.length - 1) === ' '){
      this.submitText();
      // this.textService.submitText(str).subscribe((response: any) => {
      //   console.log('Received: ' + response);
      //   this.responseText = response.response;
      // });
    }
  }

}
